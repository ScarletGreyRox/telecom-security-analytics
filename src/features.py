"""
features.py
-----------
Feature engineering for the SAS821S Capstone Project.

Builds ML-ready feature tables from raw telemetry, aggregated per host
per configurable time window (default: 1 hour).

Feature groups (each tied to a charter requirement):
  1. Beacon timing      -> regular-interval C2 detection
  2. DNS analytics      -> entropy-based tunnelling detection
  3. Byte/off-hours     -> exfiltration detection
  4. Auth analytics     -> lateral movement detection

Main entry point:
    build_features(netflow, dns, auth, window="1h") -> pd.DataFrame
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
WINDOW = "1h"                    # aggregation window
BUSINESS_HOUR_START = 8
BUSINESS_HOUR_END = 18
MIN_EVENTS = 2   # drop (host, window) rows with fewer than this many events
                 # (removes single-event noise; keeps all meaningful activity)


# ===========================================================================
# 1. BEACON TIMING FEATURES  (per host, per window)
# ===========================================================================
def _beacon_features_for_group(g: pd.DataFrame) -> pd.Series:
    """
    For a single (host, window) group of netflow events, compute features
    that reveal periodic beaconing behaviour.
    """
    out = {}

    # sort by time and compute inter-arrival deltas (seconds)
    g_sorted = g.sort_values("timestamp")
    deltas = g_sorted["timestamp"].diff().dt.total_seconds().dropna()

    if len(deltas) >= 2:
        out["delta_mean"]   = deltas.mean()
        out["delta_std"]    = deltas.std()
        out["delta_median"] = deltas.median()
        # coefficient of variation: low CV == very regular == beacon-like
        out["delta_cv"]     = (deltas.std() / deltas.mean()
                               if deltas.mean() > 0 else np.nan)
        # regularity score: 1.0 == perfectly regular, 0 == random
        # (uses median absolute deviation of deltas)
        mad = (deltas - deltas.median()).abs().median()
        out["delta_mad"]    = mad
        out["regularity"]   = 1.0 / (1.0 + mad) if mad >= 0 else np.nan
    else:
        out["delta_mean"]   = np.nan
        out["delta_std"]    = np.nan
        out["delta_median"] = np.nan
        out["delta_cv"]     = np.nan
        out["delta_mad"]    = np.nan
        out["regularity"]   = np.nan

    # how much does this host repeat the same destination IP?
    out["unique_dst_ips"]     = g["dst_ip"].nunique()
    out["top_dst_share"]      = (g["dst_ip"].value_counts().iloc[0] / len(g)
                                 if len(g) > 0 else np.nan)
    out["unique_dst_ports"]   = g["dst_port"].nunique()
    out["top_port_share"]     = (g["dst_port"].value_counts().iloc[0] / len(g)
                                 if len(g) > 0 else np.nan)

    # event volume in this window
    out["event_count"] = len(g)

    # average bytes per event (beacons are tiny)
    out["avg_bytes_sent"]  = g["bytes_sent"].mean()
    out["avg_bytes_recvd"] = g["bytes_recvd"].mean()

    return pd.Series(out)


def build_beacon_features(netflow: pd.DataFrame,
                          window: str = WINDOW) -> pd.DataFrame:
    """
    Aggregate netflow by (src_ip, time-window) and compute beacon features.

    Returns a dataframe indexed by (src_ip, window_start) with one row per host
    per window, containing all beacon-timing features.
    """
    df = netflow.copy()
    df["window_start"] = df["timestamp"].dt.floor(window)

    grouped = (
        df.groupby(["src_ip", "window_start"], group_keys=False)
          .apply(_beacon_features_for_group)
          .reset_index()
    )
    return grouped


# ===========================================================================
# 2. DNS FEATURES  (per host, per window)
# ===========================================================================
def _shannon_entropy(s: str) -> float:
    """Shannon entropy in bits per character."""
    if not s:
        return 0.0
    from collections import Counter
    counts = Counter(s)
    n = len(s)
    return -sum((c / n) * np.log2(c / n) for c in counts.values())


def _dns_features_for_group(g: pd.DataFrame) -> pd.Series:
    """Per-(host, window) DNS behaviour features."""
    out = {}

    if len(g) == 0:
        return pd.Series({
            "dns_count":              0,
            "dns_avg_sub_len":        np.nan,
            "dns_max_sub_len":        np.nan,
            "dns_avg_entropy":        np.nan,
            "dns_max_entropy":        np.nan,
            "dns_digit_ratio":        np.nan,
            "dns_unique_domains":     0,
            "dns_unique_subdomains":  0,
            "dns_subdomain_uniqueness": np.nan,
            "dns_txt_ratio":          np.nan,
            "dns_nxdomain_ratio":     np.nan,
            "dns_top_domain_share":   np.nan,
        })

    # subdomain = first label of the query
    subdomains = g["query"].astype(str).str.split(".").str[0]

    sub_lengths = subdomains.str.len()
    entropies   = subdomains.apply(_shannon_entropy)

    # digit ratio in subdomain
    digits = subdomains.str.count(r"\d")
    digit_ratio = (digits.sum() / sub_lengths.sum()
                   if sub_lengths.sum() > 0 else np.nan)

    out["dns_count"]              = len(g)
    out["dns_avg_sub_len"]        = sub_lengths.mean()
    out["dns_max_sub_len"]        = sub_lengths.max()
    out["dns_avg_entropy"]        = entropies.mean()
    out["dns_max_entropy"]        = entropies.max()
    out["dns_digit_ratio"]        = digit_ratio
    out["dns_unique_domains"]     = g["domain"].nunique()
    out["dns_unique_subdomains"]  = subdomains.nunique()
    out["dns_subdomain_uniqueness"] = (
        subdomains.nunique() / len(subdomains) if len(subdomains) > 0 else np.nan
    )
    out["dns_txt_ratio"]          = (
        (g["query_type"] == "TXT").mean() if "query_type" in g.columns else np.nan
    )
    out["dns_nxdomain_ratio"]     = (
        (g["response_code"] == "NXDOMAIN").mean()
        if "response_code" in g.columns else np.nan
    )
    out["dns_top_domain_share"]   = (
        g["domain"].value_counts().iloc[0] / len(g) if len(g) > 0 else np.nan
    )

    return pd.Series(out)


def build_dns_features(dns: pd.DataFrame,
                       window: str = WINDOW) -> pd.DataFrame:
    """Aggregate DNS logs by (src_ip, window) with entropy/length features."""
    df = dns.copy()
    df["window_start"] = df["timestamp"].dt.floor(window)

    grouped = (
        df.groupby(["src_ip", "window_start"], group_keys=False)
          .apply(_dns_features_for_group)
          .reset_index()
    )
    return grouped


# ===========================================================================
# 3. BYTE / OFF-HOURS FEATURES  (per host, per window)
# ===========================================================================
def _byte_features_for_group(g: pd.DataFrame) -> pd.Series:
    """Per-(host, window) byte-behaviour and time-of-day features."""
    out = {}

    if len(g) == 0:
        return pd.Series({
            "byte_total_sent":      0,
            "byte_total_recvd":     0,
            "byte_sent_recv_ratio": np.nan,
            "byte_avg_sent":        np.nan,
            "byte_max_sent":        np.nan,
            "byte_log_max_sent":    np.nan,
            "off_hours_ratio":      np.nan,
            "night_events":         0,
        })

    sent  = g["bytes_sent"]
    recvd = g["bytes_recvd"]
    total_sent  = sent.sum()
    total_recvd = recvd.sum()

    out["byte_total_sent"]       = total_sent
    out["byte_total_recvd"]      = total_recvd
    out["byte_sent_recv_ratio"]  = (
        total_sent / total_recvd if total_recvd > 0 else np.nan
    )
    out["byte_avg_sent"]         = sent.mean()
    out["byte_max_sent"]         = sent.max()
    # log transform to compress the huge exfiltration outliers
    out["byte_log_max_sent"]     = np.log10(sent.max() + 1)

    # how many events fell outside business hours?
    hours = g["timestamp"].dt.hour
    off_hours_mask = (hours < BUSINESS_HOUR_START) | (hours >= BUSINESS_HOUR_END)
    out["off_hours_ratio"] = off_hours_mask.mean()
    out["night_events"]    = int(off_hours_mask.sum())

    return pd.Series(out)


def build_byte_features(netflow: pd.DataFrame,
                        window: str = WINDOW) -> pd.DataFrame:
    """Aggregate netflow by (src_ip, window) with byte and off-hours features."""
    df = netflow.copy()
    df["window_start"] = df["timestamp"].dt.floor(window)

    grouped = (
        df.groupby(["src_ip", "window_start"], group_keys=False)
          .apply(_byte_features_for_group)
          .reset_index()
    )
    return grouped


# ===========================================================================
# 4. AUTH / LATERAL-MOVEMENT FEATURES  (per user, per window)
# ===========================================================================
def _auth_features_for_group(g: pd.DataFrame) -> pd.Series:
    """Per-(user, window) authentication behaviour features."""
    out = {}

    if len(g) == 0:
        return pd.Series({
            "auth_count":           0,
            "auth_failure_ratio":   np.nan,
            "auth_unique_hosts":    0,
            "auth_unique_src_ips":  0,
            "auth_target_rate":     np.nan,
            "auth_off_hours_ratio": np.nan,
        })

    out["auth_count"]        = len(g)
    out["auth_failure_ratio"] = (g["status"] == "FAILURE").mean()
    out["auth_unique_hosts"]  = g["host"].nunique()
    out["auth_unique_src_ips"] = g["src_ip"].nunique()

    # target rate: many distinct hosts accessed in a short window
    # is characteristic of lateral movement
    out["auth_target_rate"] = (
        g["host"].nunique() / len(g) if len(g) > 0 else np.nan
    )

    hours = g["timestamp"].dt.hour
    off_hours_mask = (hours < BUSINESS_HOUR_START) | (hours >= BUSINESS_HOUR_END)
    out["auth_off_hours_ratio"] = off_hours_mask.mean()

    return pd.Series(out)


def build_auth_features(auth: pd.DataFrame,
                        window: str = WINDOW) -> pd.DataFrame:
    """Aggregate auth logs by (user, window) with lateral-movement features."""
    df = auth.copy()
    df["window_start"] = df["timestamp"].dt.floor(window)

    grouped = (
        df.groupby(["user", "window_start"], group_keys=False)
          .apply(_auth_features_for_group)
          .reset_index()
    )
    return grouped


# ===========================================================================
# 5. MASTER FEATURE BUILDER
# ===========================================================================
def build_features(netflow: pd.DataFrame,
                   dns: pd.DataFrame,
                   auth: pd.DataFrame,
                   window: str = WINDOW) -> pd.DataFrame:
    """
    Build the unified ML-ready feature table.

    Each row is one (entity, window) pair where entity is either:
      - 'host:<ip>'  for netflow / DNS-derived features
      - 'user:<uid>' for auth-derived features

    All feature groups are joined on the (entity, window_start) key so that
    the ML models see a single wide table.
    """
    print("Building beacon features ...")
    beacon = build_beacon_features(netflow, window)
    beacon = beacon.rename(columns={"src_ip": "entity"})
    beacon["entity"] = "host:" + beacon["entity"].astype(str)

    print("Building byte features ...")
    byte = build_byte_features(netflow, window)
    byte = byte.rename(columns={"src_ip": "entity"})
    byte["entity"] = "host:" + byte["entity"].astype(str)

    print("Building DNS features ...")
    dnsf = build_dns_features(dns, window)
    dnsf = dnsf.rename(columns={"src_ip": "entity"})
    dnsf["entity"] = "host:" + dnsf["entity"].astype(str)

    print("Building auth features ...")
    authf = build_auth_features(auth, window)
    authf = authf.rename(columns={"user": "entity"})
    authf["entity"] = "user:" + authf["entity"].astype(str)

    # ---- Merge host-side features (beacon + byte + DNS) ----
    print("Merging host-side features ...")
    host_feats = beacon.merge(
        byte, on=["entity", "window_start"], how="outer", suffixes=("", "_byte")
    )
    host_feats = host_feats.merge(
        dnsf, on=["entity", "window_start"], how="outer", suffixes=("", "_dns")
    )

    # ---- Merge auth-side features ----
    print("Merging auth-side features ...")
    all_feats = pd.concat([host_feats, authf], ignore_index=True,
                          sort=False)

    # ---- Ground-truth label ----
    # Mark known compromised hosts from the generator's scenario
    COMPROMISED_HOSTS = {"10.0.1.50", "10.0.2.77", "10.0.3.111"}
    all_feats["is_compromised_host"] = all_feats["entity"].apply(
        lambda x: 1 if x.startswith("host:") and x.split(":", 1)[1]
                    in COMPROMISED_HOSTS else 0
    )

    # ---- Time-of-day column for later analysis ----
    all_feats["hour_of_day"] = all_feats["window_start"].dt.hour
    all_feats["day_of_week"] = all_feats["window_start"].dt.dayofweek

    # ---- Drop rows where everything is NaN (empty windows) ----
    feature_cols = [c for c in all_feats.columns
                    if c not in ("entity", "window_start",
                                 "is_compromised_host",
                                 "hour_of_day", "day_of_week")]
    all_feats = all_feats.dropna(subset=feature_cols, how="all").reset_index(drop=True)

    rows_before = len(all_feats)

    # Filter: keep windows where at least one telemetry source has
    # >= MIN_EVENTS of activity. A single sparse signal in each source
    # is still noise, so we use OR (not SUM) here.
    keep = (
        (all_feats["event_count"].fillna(0) >= MIN_EVENTS) |
        (all_feats["dns_count"].fillna(0)    >= MIN_EVENTS) |
        (all_feats["auth_count"].fillna(0)   >= MIN_EVENTS)
    )
    all_feats = all_feats[keep].reset_index(drop=True)

    rows_after = len(all_feats)

    print(f"Feature table built: {rows_before:,} rows  ->  "
          f"kept {rows_after:,} after MIN_EVENTS>={MIN_EVENTS} filter")
    print(f"Feature table shape: {all_feats.shape[0]:,} rows × {all_feats.shape[1]} cols")
    return all_feats


if __name__ == "__main__":
    import os
    RAW = "../data/raw"
    OUT = "../data/processed"
    os.makedirs(OUT, exist_ok=True)

    print("Loading raw telemetry ...")
    netflow = pd.read_csv(os.path.join(RAW, "netflow.csv"),
                          parse_dates=["timestamp"])
    dns = pd.read_csv(os.path.join(RAW, "dns.csv"),
                      parse_dates=["timestamp"])
    auth = pd.read_csv(os.path.join(RAW, "auth.csv"),
                       parse_dates=["timestamp"])

    feats = build_features(netflow, dns, auth)
    out_path = os.path.join(OUT, "features.parquet")
    feats.to_parquet(out_path, index=False)
    print(f"✔ Saved: {out_path}")

    # Quick summary
    print()
    print("=" * 60)
    print("Feature table summary")
    print("=" * 60)
    print(f"Rows:                 {len(feats):,}")
    print(f"Entities:             {feats['entity'].nunique():,}")
    print(f"Compromised windows:  {feats['is_compromised_host'].sum():,}")
    print(f"Feature columns:      {feats.shape[1] - 5}")
