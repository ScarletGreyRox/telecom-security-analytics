"""
data_generator.py
-----------------
Synthetic multi-source telemetry generator for the SAS821S Capstone Project:
"Multi-Source Telemetry Analytics for Detecting Command-and-Control Beacons
and Lateral Movement in Telecom Networks"

Produces:
  - data/raw/netflow.csv   : realistic network flow records (multi-day, diurnal)
  - data/raw/dns.csv       : DNS query logs (benign + tunnelling)
  - data/raw/auth.csv      : authentication / endpoint logs (incl. lateral movement)

Attack scenarios injectable (per charter):
  - C2 beaconing (periodic, jittered)
  - DNS tunnelling (high-entropy subdomains)
  - Off-hours data exfiltration (large outbound transfers at night)
  - Lateral movement (authentication + SMB/RDP chains)

Run standalone:
  python data_generator.py
"""

from __future__ import annotations

import os
import random
import string
import ipaddress
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------------
CONFIG = {
    "start_date":        "2026-08-10 00:00:00",   # Monday
    "days":              7,                        # one full week
    "internal_subnets":  ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"],
    "external_subnet":   "172.16.0.0/16",
    "business_hour_start": 8,     # 08:00
    "business_hour_end":  18,     # 18:00
    "night_factor":      0.15,    # 15% of business-hours volume at night
    "peak_factor":       1.60,    # lunch/afternoon peak multiplier
    "base_events_per_minute": 12,  # mean events/min during business hours
}

# ---------------------------------------------------------------------------
# Realistic service catalogue: (port, protocol, typical_bytes_sent_range,
# typical_bytes_recvd_range, weight)
# ---------------------------------------------------------------------------
SERVICES = [
    # port, proto, sent_bytes_low, sent_high, recv_low, recv_high, weight
    (443,  "TCP",  2_000,   80_000,  10_000, 1_500_000, 40),   # HTTPS
    (80,   "TCP",  1_000,   30_000,   5_000,   800_000, 15),   # HTTP
    (53,   "UDP",    100,      500,     200,     2_000, 12),   # DNS
    (22,   "TCP",    500,    5_000,     500,    10_000,  8),   # SSH
    (3389, "TCP",  1_000,   20_000,   2_000,    60_000,  5),   # RDP
    (25,   "TCP",    500,   10_000,     500,     5_000,  4),   # SMTP
    (110,  "TCP",    200,    3_000,     300,     4_000,  3),   # POP3
    (143,  "TCP",    200,    3_000,     300,     4_000,  3),   # IMAP
    (445,  "TCP",  2_000,   50_000,   5_000,   120_000,  4),   # SMB
    (123,  "UDP",     50,      200,      50,       200,  3),   # NTP
    (67,   "UDP",     50,      300,      50,       300,  1),   # DHCP
    (161,  "UDP",    100,      800,     200,     1_500,  1),   # SNMP
    (8080, "TCP",  1_000,   40_000,   3_000,   400_000,  1),   # HTTP-alt
]

_ports    = [s[0] for s in SERVICES]
_protos   = [s[1] for s in SERVICES]
_weights  = np.array([s[6] for s in SERVICES], dtype=float)
_weights /= _weights.sum()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _random_internal_ip() -> str:
    subnet = random.choice(CONFIG["internal_subnets"])
    net = ipaddress.ip_network(subnet)
    return str(net.network_address + random.randint(2, net.num_addresses - 2))

def _random_external_ip() -> str:
    net = ipaddress.ip_network(CONFIG["external_subnet"])
    return str(net.network_address + random.randint(2, net.num_addresses - 2))

def _random_ephemeral_port() -> int:
    return random.randint(32768, 60999)

def _pick_service():
    idx = np.random.choice(len(SERVICES), p=_weights)
    port, proto, s_lo, s_hi, r_lo, r_hi, _ = SERVICES[idx]
    sent = int(np.random.lognormal(
        mean=np.log((s_lo + s_hi) / 2), sigma=0.6))
    recv = int(np.random.lognormal(
        mean=np.log((r_lo + r_hi) / 2), sigma=0.7))
    sent = max(s_lo, min(s_hi, sent))
    recv = max(r_lo, min(r_hi, recv))
    return port, proto, sent, recv

def _hourly_rate_multiplier(hour: int, weekday: int = 0) -> float:
    """
    Diurnal pattern:
      - high during business hours, low at night
      - lunch-time dip
      - afternoon peak
      - weekends are quieter (~40% of weekday business-hours volume)
    weekday: 0=Monday ... 6=Sunday
    """
    # Weekend reduction factor
    weekend_factor = 0.4 if weekday >= 5 else 1.0

    if CONFIG["business_hour_start"] <= hour < CONFIG["business_hour_end"]:
        if hour in (12, 13):
            base = CONFIG["peak_factor"] * 0.8   # lunch dip
        elif 14 <= hour <= 16:
            base = CONFIG["peak_factor"]        # afternoon peak
        else:
            base = 1.0
        return base * weekend_factor

    # Night: weekends are a bit busier at night (social traffic)
    night_mult = CONFIG["night_factor"] * (1.5 if weekday >= 5 else 1.0)
    return night_mult

def _random_string(n: int) -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=n))


# ===========================================================================
# 1. NETFLOW GENERATOR
# ===========================================================================
def generate_netflow() -> pd.DataFrame:
    """Generate realistic multi-day NetFlow records with diurnal variation."""
    start = datetime.strptime(CONFIG["start_date"], "%Y-%m-%d %H:%M:%S")
    days = CONFIG["days"]
    base = CONFIG["base_events_per_minute"]

    rows: List[dict] = []
    for day in range(days):
        for hour in range(24):
            # day is 0-based from start_date; start_date is a Monday, so weekday = day % 7
            weekday = day % 7
            multiplier = _hourly_rate_multiplier(hour, weekday)
            for minute in range(60):
                # Poisson-jittered event count per minute
                mean_events = base * multiplier
                n_events = max(1, int(np.random.poisson(mean_events)))

                for _ in range(n_events):
                    ts = start + timedelta(
                        days=day, hours=hour, minutes=minute,
                        seconds=random.randint(0, 59))

                    port, proto, sent, recv = _pick_service()

                    # small chance of DENY (blocked traffic)
                    action = "DENY" if random.random() < 0.03 else "ALLOW"

                    rows.append({
                        "timestamp":   ts,
                        "src_ip":      _random_internal_ip(),
                        "dst_ip":      _random_external_ip(),
                        "src_port":    _random_ephemeral_port(),
                        "dst_port":    port,
                        "bytes_sent":  sent,
                        "bytes_recvd": recv,
                        "protocol":    proto,
                        "action":      action,
                    })

    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    return df


# ===========================================================================
# 2. DNS LOG GENERATOR
# ===========================================================================
BENIGN_DOMAINS = [
    "google.com", "microsoft.com", "office365.com", "outlook.com",
    "cloudflare.com", "amazonaws.com", "azure.com", "github.com",
    "slack.com", "zoom.us", "salesforce.com", "oracle.com",
    "wikipedia.org", "linkedin.com", "netflix.com", "akamai.net",
    "youtube.com", "cdn.jsdelivr.net", "gstatic.com", "telecom.local",
]

def _benign_query() -> Tuple[str, str]:
    base = random.choice(BENIGN_DOMAINS)
    # occasionally add a subdomain
    if random.random() < 0.4:
        sub = random.choice(["www", "api", "cdn", "mail", "auth",
                             "portal", "static", "login", "app", "vpn"])
        return f"{sub}.{base}", base
    return base, base

def generate_dns(n_per_minute: int = 6) -> pd.DataFrame:
    """Generate DNS query logs with variable rate."""
    start = datetime.strptime(CONFIG["start_date"], "%Y-%m-%d %H:%M:%S")
    days = CONFIG["days"]

    rows: List[dict] = []
    for day in range(days):
        for hour in range(24):
            # day is 0-based from start_date; start_date is a Monday, so weekday = day % 7
            weekday = day % 7
            multiplier = _hourly_rate_multiplier(hour, weekday)
            for minute in range(60):
                n = max(1, int(np.random.poisson(n_per_minute * multiplier)))
                for _ in range(n):
                    ts = start + timedelta(
                        days=day, hours=hour, minutes=minute,
                        seconds=random.randint(0, 59))
                    query, base = _benign_query()
                    rows.append({
                        "timestamp":      ts,
                        "src_ip":         _random_internal_ip(),
                        "query":          query,
                        "query_type":     random.choices(
                                            ["A", "AAAA", "CNAME", "MX", "TXT"],
                                            weights=[70, 15, 8, 4, 3])[0],
                        "response_code":  "NOERROR" if random.random() < 0.95
                                          else "NXDOMAIN",
                        "response_ip":    _random_external_ip()
                                          if random.random() < 0.8 else None,
                        "domain":         base,
                    })

    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)


# ===========================================================================
# 3. AUTHENTICATION / ENDPOINT LOG GENERATOR
# ===========================================================================
USERS = [f"user{ i:03d}" for i in range(1, 61)]   # 60 users

def generate_auth(n_per_minute: int = 2) -> pd.DataFrame:
    """Generate authentication logs (login success/failure)."""
    start = datetime.strptime(CONFIG["start_date"], "%Y-%m-%d %H:%M:%S")
    days = CONFIG["days"]

    rows: List[dict] = []
    for day in range(days):
        for hour in range(24):
            # day is 0-based from start_date; start_date is a Monday, so weekday = day % 7
            weekday = day % 7
            multiplier = _hourly_rate_multiplier(hour, weekday)
            for minute in range(60):
                n = max(0, int(np.random.poisson(n_per_minute * multiplier)))
                for _ in range(n):
                    ts = start + timedelta(
                        days=day, hours=hour, minutes=minute,
                        seconds=random.randint(0, 59))
                    user = random.choice(USERS)
                    success = random.random() > 0.08   # ~8% failures
                    rows.append({
                        "timestamp":   ts,
                        "user":        user,
                        "src_ip":      _random_internal_ip(),
                        "host":        f"ws-{random.randint(1, 200):03d}",
                        "event_type":  "LOGIN",
                        "status":      "SUCCESS" if success else "FAILURE",
                        "auth_method": random.choice(
                                          ["PASSWORD", "KERBEROS", "TOKEN"]),
                    })

    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)


# ===========================================================================
# 4. ATTACK INJECTION FUNCTIONS  (per charter requirements)
# ===========================================================================
MALICIOUS_DOMAINS = [
    "evil-c2.example",
    "badactor-update.net",
    "dns-tunnel.xyz",
    "malware-cdn.top",
    "apt-relay.cc",
]
MALICIOUS_IPS = [
    "172.16.99.10", "172.16.99.20", "172.16.99.30",
    "172.16.99.40", "172.16.99.50",
]

# Chosen compromised internal hosts for the scenario
COMPROMISED_HOSTS = ["10.0.1.50", "10.0.2.77", "10.0.3.111"]
BEACON_TARGET    = "172.16.99.10"
EXFIL_TARGET     = "172.16.99.20"


def inject_c2_beacon(netflow: pd.DataFrame,
                     dns: pd.DataFrame,
                     beacon_interval_sec: int = 60,
                     jitter_sec: int = 5,
                     duration_hours: int = 6,
                     start_hour_of_day: int = 10) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Inject a C2 beacon pattern:
      - Compromised host contacts MALICIOUS_IPS periodically (regular interval, small jitter)
      - Same host makes periodic DNS queries to a malicious domain
    """
    start = datetime.strptime(CONFIG["start_date"], "%Y-%m-%d %H:%M:%S")
    beacon_host = COMPROMISED_HOSTS[0]
    beacon_domain = MALICIOUS_DOMAINS[0]

    rows_net, rows_dns = [], []
    for day in range(CONFIG["days"]):
        day_start = start + timedelta(days=day, hours=start_hour_of_day)
        end = day_start + timedelta(hours=duration_hours)

        t = day_start
        while t < end:
            # small timing jitter to make detection non-trivial
            jitter = random.randint(-jitter_sec, jitter_sec)
            ts = t + timedelta(seconds=jitter)

            rows_net.append({
                "timestamp":   ts,
                "src_ip":      beacon_host,
                "dst_ip":      BEACON_TARGET,
                "src_port":    _random_ephemeral_port(),
                "dst_port":    443,
                "bytes_sent":  random.randint(200, 800),    # small
                "bytes_recvd": random.randint(150, 600),    # small
                "protocol":    "TCP",
                "action":      "ALLOW",
            })
            rows_dns.append({
                "timestamp":     ts,
                "src_ip":        beacon_host,
                "query":         f"beacon.{beacon_domain}",
                "query_type":    "A",
                "response_code": "NOERROR",
                "response_ip":   BEACON_TARGET,
                "domain":        beacon_domain,
            })

            t += timedelta(seconds=beacon_interval_sec)

    netflow = pd.concat([netflow, pd.DataFrame(rows_net)], ignore_index=True)
    dns = pd.concat([dns, pd.DataFrame(rows_dns)], ignore_index=True)
    return netflow, dns


def inject_dns_tunnelling(dns: pd.DataFrame,
                          n_queries_per_hour: int = 40,
                          duration_hours: int = 5,
                          start_hour_of_day: int = 9) -> pd.DataFrame:
    """
    Inject DNS tunnelling:
      - High-entropy, long, random subdomains under a malicious domain
      - Many TXT queries (data encoded in responses)
    """
    start = datetime.strptime(CONFIG["start_date"], "%Y-%m-%d %H:%M:%S")
    tunnel_host = COMPROMISED_HOSTS[1]
    tunnel_domain = MALICIOUS_DOMAINS[2]

    rows = []
    for day in range(CONFIG["days"]):
        day_start = start + timedelta(days=day, hours=start_hour_of_day)
        end = day_start + timedelta(hours=duration_hours)

        n_total = n_queries_per_hour * duration_hours
        for _ in range(n_total):
            ts = day_start + timedelta(
                seconds=random.randint(0, duration_hours * 3600))
            sub = _random_string(random.randint(20, 40))
            rows.append({
                "timestamp":     ts,
                "src_ip":        tunnel_host,
                "query":         f"{sub}.{tunnel_domain}",
                "query_type":    random.choices(["TXT", "A", "AAAA"],
                                                weights=[60, 30, 10])[0],
                "response_code": "NOERROR",
                "response_ip":   None,
                "domain":        tunnel_domain,
            })

    dns = pd.concat([dns, pd.DataFrame(rows)], ignore_index=True)
    return dns


def inject_exfiltration(netflow: pd.DataFrame,
                        n_transfers_per_night: int = 6,
                        duration_nights: int = 5,
                        start_hour: int = 2) -> pd.DataFrame:
    """
    Inject off-hours data exfiltration:
      - Large outbound transfers at 02:00-03:00 (well outside business hours)
      - Asymmetric bytes (sent >> received)
    """
    start = datetime.strptime(CONFIG["start_date"], "%Y-%m-%d %H:%M:%S")
    exfil_host = COMPROMISED_HOSTS[2]

    rows = []
    for day in range(duration_nights):
        for _ in range(n_transfers_per_night):
            ts = start + timedelta(
                days=day, hours=start_hour,
                minutes=random.randint(0, 59),
                seconds=random.randint(0, 59))
            sent = random.randint(5_000_000, 50_000_000)  # 5-50 MB out
            recv = random.randint(1_000, 20_000)          # tiny response
            rows.append({
                "timestamp":   ts,
                "src_ip":      exfil_host,
                "dst_ip":      EXFIL_TARGET,
                "src_port":    _random_ephemeral_port(),
                "dst_port":    443,
                "bytes_sent":  sent,
                "bytes_recvd": recv,
                "protocol":    "TCP",
                "action":      "ALLOW",
            })

    netflow = pd.concat([netflow, pd.DataFrame(rows)], ignore_index=True)
    return netflow


def inject_lateral_movement(netflow: pd.DataFrame,
                            auth: pd.DataFrame,
                            chain_length: int = 4) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Inject lateral movement:
      - One host authenticates to several internal hosts in quick succession
      - Followed by SMB (445) traffic to those same hosts
    """
    start = datetime.strptime(CONFIG["start_date"], "%Y-%m-%d %H:%M:%S")
    pivot = COMPROMISED_HOSTS[0]
    user  = "user007"

    rows_net, rows_auth = [], []
    for day in range(CONFIG["days"]):
        base_ts = start + timedelta(days=day, hours=14, minutes=30)
        targets = random.sample(
            [f"10.0.{random.randint(1,3)}.{random.randint(1,254)}"
             for _ in range(chain_length * 2)],
            chain_length)

        for i, target in enumerate(targets):
            ts_auth = base_ts + timedelta(minutes=i * 3)
            ts_net  = ts_auth + timedelta(seconds=random.randint(10, 60))

            rows_auth.append({
                "timestamp":   ts_auth,
                "user":        user,
                "src_ip":      pivot,
                "host":        f"ws-{random.randint(1, 200):03d}",
                "event_type":  "LOGIN",
                "status":      "SUCCESS",
                "auth_method": "KERBEROS",
            })
            rows_net.append({
                "timestamp":   ts_net,
                "src_ip":      pivot,
                "dst_ip":      target,
                "src_port":    _random_ephemeral_port(),
                "dst_port":    445,
                "bytes_sent":  random.randint(5_000, 50_000),
                "bytes_recvd": random.randint(2_000, 30_000),
                "protocol":    "TCP",
                "action":      "ALLOW",
            })

    netflow = pd.concat([netflow, pd.DataFrame(rows_net)], ignore_index=True)
    auth    = pd.concat([auth,    pd.DataFrame(rows_auth)], ignore_index=True)
    return netflow, auth


# ===========================================================================
# 5. ORCHESTRATION + SAVE
# ===========================================================================
def save_all(netflow: pd.DataFrame,
             dns: pd.DataFrame,
             auth: pd.DataFrame,
             out_dir: str = "../data/raw") -> None:
    os.makedirs(out_dir, exist_ok=True)
    netflow.sort_values("timestamp").to_csv(
        os.path.join(out_dir, "netflow.csv"), index=False)
    dns.sort_values("timestamp").to_csv(
        os.path.join(out_dir, "dns.csv"), index=False)
    auth.sort_values("timestamp").to_csv(
        os.path.join(out_dir, "auth.csv"), index=False)
    print(f"✔ netflow.csv : {len(netflow):>7,} rows")
    print(f"✔ dns.csv     : {len(dns):>7,} rows")
    print(f"✔ auth.csv    : {len(auth):>7,} rows")


def build_dataset(inject_attacks: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("Generating NetFlow ...")
    netflow = generate_netflow()
    print("Generating DNS ...")
    dns = generate_dns()
    print("Generating Auth ...")
    auth = generate_auth()

    if inject_attacks:
        print("Injecting C2 beacon ...")
        netflow, dns = inject_c2_beacon(netflow, dns)
        print("Injecting DNS tunnelling ...")
        dns = inject_dns_tunnelling(dns)
        print("Injecting exfiltration ...")
        netflow = inject_exfiltration(netflow)
        print("Injecting lateral movement ...")
        netflow, auth = inject_lateral_movement(netflow, auth)

    return netflow, dns, auth


if __name__ == "__main__":
    netflow, dns, auth = build_dataset(inject_attacks=True)
    save_all(netflow, dns, auth)
    print("\nDone. Data written to data/raw/")
