"""
auth_analytics.py
-----------------
Authentication analytics and lateral-movement detection.

Detects the "credential pivot" pattern:
    1. A user logs into several distinct internal hosts in a short window
    2. SMB (445) / RDP (3389) traffic follows to those same hosts
    3. All within a tight temporal chain (minutes, not hours)

Outputs:
    reports/auth_alerts.csv          per-window risk flags
    reports/lateral_incidents.json   assembled incident chains
    reports/auth_summary.json        aggregate metrics
"""

from __future__ import annotations

import os
import json
import numpy as np
import pandas as pd
from datetime import timedelta

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
WINDOW = "15min"                # tight windows — lateral movement is fast
FANOUT_MIN_HOSTS = 3            # at least 3 distinct hosts
FANOUT_TARGET_RATE_MIN = 0.5    # distinct hosts / total logins
SMB_PORTS = {445, 3389}         # SMB + RDP
CHAIN_GAP_MINUTES = 20          # events within this gap belong to same chain


# ===========================================================================
# 1. PER-WINDOW DETECTION  (user x 15-min window)
# ===========================================================================
def _per_window_auth_features(g: pd.DataFrame) -> pd.Series:
    """Compute fan-out features for a single (user, window) group."""
    if len(g) == 0:
        return pd.Series({
            "auth_count": 0,
            "unique_hosts": 0,
            "unique_src_ips": 0,
            "failure_ratio": np.nan,
            "target_rate": np.nan,
            "kerberos_ratio": np.nan,
            "off_hours": 0,
        })

    hosts = g["host"].nunique()
    count = len(g)

    out = {
        "auth_count":     count,
        "unique_hosts":   hosts,
        "unique_src_ips": g["src_ip"].nunique(),
        "failure_ratio":  (g["status"] == "FAILURE").mean(),
        "target_rate":    hosts / count if count else np.nan,
        "kerberos_ratio": (g["auth_method"] == "KERBEROS").mean()
                           if "auth_method" in g.columns else np.nan,
    }

    hours = g["timestamp"].dt.hour
    out["off_hours"] = int(((hours < 8) | (hours >= 18)).sum())
    return pd.Series(out)


def detect_auth_anomalies(auth: pd.DataFrame,
                          window: str = WINDOW) -> pd.DataFrame:
    """
    Produce per-(user, window) auth anomaly flags.

    A window is flagged when:
        unique_hosts >= FANOUT_MIN_HOSTS  AND
        target_rate   >= FANOUT_TARGET_RATE_MIN
    """
    df = auth.copy()
    df["window_start"] = df["timestamp"].dt.floor(window)

    feats = (
        df.groupby(["user", "window_start"], group_keys=False)
          .apply(_per_window_auth_features)
          .reset_index()
    )

    # Rule-based anomaly flag
    feats["is_fanout"] = (
        (feats["unique_hosts"] >= FANOUT_MIN_HOSTS) &
        (feats["target_rate"]  >= FANOUT_TARGET_RATE_MIN)
    ).astype(int)

    return feats


# ===========================================================================
# 2. INCIDENT CHAIN ASSEMBLY
# ===========================================================================
def build_lateral_incidents(auth_feats: pd.DataFrame,
                            netflow: pd.DataFrame,
                            auth_raw: pd.DataFrame,
                            gap_minutes: int = CHAIN_GAP_MINUTES) -> list:
    """
    Assemble flagged auth windows into coherent lateral-movement incidents.

    An incident is a *chronological chain* of:
      - authentication events (fan-out to multiple hosts)
      - followed by SMB / RDP traffic to those same hosts

    Multiple flagged windows for the same user, close in time, are merged
    into one incident chain.
    """
    flagged = auth_feats[auth_feats["is_fanout"] == 1].copy()
    flagged = flagged.sort_values(["user", "window_start"]).reset_index(drop=True)

    if len(flagged) == 0:
        return []

    # --- Group flagged windows into chains: same user + gaps < gap_minutes ---
    incidents = []
    gap = pd.Timedelta(minutes=gap_minutes)

    for user, group in flagged.groupby("user"):
        group = group.sort_values("window_start").reset_index(drop=True)

        current_chain = [group.iloc[0]]
        for i in range(1, len(group)):
            prev_end = current_chain[-1]["window_start"] + pd.Timedelta(minutes=15)
            if group.iloc[i]["window_start"] - prev_end <= gap:
                current_chain.append(group.iloc[i])
            else:
                incidents.append((user, current_chain))
                current_chain = [group.iloc[i]]
        incidents.append((user, current_chain))

    # --- For each chain, build the incident record ---
    output = []
    netflow_ts = netflow.copy()
    netflow_ts["timestamp"] = pd.to_datetime(netflow_ts["timestamp"])

    for user, chain in incidents:
        chain_df = pd.DataFrame(chain)
        start = chain_df["window_start"].min()
        end   = chain_df["window_start"].max() + pd.Timedelta(minutes=15)

        total_logins = int(chain_df["auth_count"].sum())
        max_fanout   = int(chain_df["unique_hosts"].max())
        avg_target_rate = float(chain_df["target_rate"].mean())
        total_off_hours = int(chain_df["off_hours"].sum())

        # --- Cross-reference with netflow: SMB / RDP ---
        #
        # IMPORTANT: we filter to SMB traffic whose *source* is one of the
        # hosts the user actually logged in from during this chain. Without
        # this filter, we'd match every SMB flow in the network during the
        # window (hundreds of hosts), which inflates the incident.
        window_netflow = netflow_ts[
            (netflow_ts["timestamp"] >= start) &
            (netflow_ts["timestamp"] <  end)
        ]
        # Pivot hosts: source IPs observed in the auth chain
        chain_auth = auth_raw[
            (auth_raw["user"] == user) &
            (auth_raw["timestamp"] >= start) &
            (auth_raw["timestamp"] <  end)
        ]
        pivot_ips = set(chain_auth["src_ip"].unique().tolist())
        auth_target_hosts = set(chain_auth["host"].unique().tolist())

        smb_events = window_netflow[
            (window_netflow["dst_port"].isin(SMB_PORTS)) &
            (
                window_netflow["src_ip"].isin(pivot_ips)
                # keep traffic only from the pivot host(s)
            )
        ]
        smb_targets = sorted(smb_events["dst_ip"].unique().tolist())

        # --- Risk score (0-100) ---
        # Weighted: fanout dominates, SMB presence corroborates
        risk = (
            40 * min(max_fanout / 6, 1.0)          # 40 pts for fanout
            + 30 * min(len(smb_targets) / 4, 1.0)  # 30 pts for SMB corroboration
            + 15 * min(total_logins / 10, 1.0)     # 15 pts for volume
            + 15 * (1.0 if total_off_hours > 0 else 0.0)  # 15 pts if off-hours
        )
        risk_score = round(risk, 1)

        output.append({
            "incident_id":     f"LM-{user}-{start.strftime('%Y%m%dT%H%M')}",
            "user":            user,
            "start":           start.isoformat(),
            "end":             end.isoformat(),
            "pivot_hosts":     sorted(pivot_ips),
            "duration_min":    int((end - start).total_seconds() // 60),
            "windows_in_chain": int(len(chain_df)),
            "total_logins":    total_logins,
            "max_fanout":      max_fanout,
            "avg_target_rate": round(avg_target_rate, 3),
            "smb_targets":     smb_targets,
            "smb_target_count": len(smb_targets),
            "off_hours_events": total_off_hours,
            "risk_score":      risk_score,
        })

    # Sort by risk descending
    output.sort(key=lambda x: -x["risk_score"])
    return output


# ===========================================================================
# 3. SUMMARY METRICS + SAVE
# ===========================================================================
def summarize(auth_feats: pd.DataFrame,
              incidents: list,
              auth_raw: pd.DataFrame) -> dict:
    """Aggregate metrics suitable for the report."""
    n_users = int(auth_raw["user"].nunique())
    n_windows = int(len(auth_feats))
    n_fanout_windows = int(auth_feats["is_fanout"].sum())
    n_incidents = len(incidents)
    high_risk = [i for i in incidents if i["risk_score"] >= 60]

    # Which users appear in the top incidents?
    top_users = [i["user"] for i in incidents[:5]]

    return {
        "n_users":            n_users,
        "n_windows":          n_windows,
        "n_fanout_windows":   n_fanout_windows,
        "n_incidents":        n_incidents,
        "n_high_risk":        len(high_risk),
        "top_incident_users": top_users,
        "max_risk_score":     max((i["risk_score"] for i in incidents),
                                   default=0.0),
    }


def save_artifacts(auth_feats: pd.DataFrame,
                   incidents: list,
                   summary: dict,
                   report_dir: str = "../reports") -> None:
    os.makedirs(report_dir, exist_ok=True)

    # Per-window flags
    flags_path = os.path.join(report_dir, "auth_alerts.csv")
    auth_feats.to_csv(flags_path, index=False)
    print(f"✔ Auth per-window flags saved: {flags_path}")

    # Incidents as JSON
    incidents_path = os.path.join(report_dir, "lateral_incidents.json")
    with open(incidents_path, "w", encoding="utf-8") as f:
        json.dump(incidents, f, indent=2)
    print(f"✔ Incidents saved: {incidents_path}")

    # Summary metrics
    summary_path = os.path.join(report_dir, "auth_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"✔ Summary saved: {summary_path}")


# ===========================================================================
# 4. MAIN PIPELINE
# ===========================================================================
def run_auth_analytics(auth: pd.DataFrame,
                       netflow: pd.DataFrame) -> dict:
    """End-to-end auth analytics: detect, chain, score, summarize."""
    print("Detecting per-window auth anomalies ...")
    auth_feats = detect_auth_anomalies(auth, window=WINDOW)
    print(f"  user-windows: {len(auth_feats):,}")
    print(f"  flagged fan-out windows: {int(auth_feats['is_fanout'].sum()):,}")

    print("\nAssembling incident chains ...")
    incidents = build_lateral_incidents(auth_feats, netflow, auth)
    print(f"  incidents found: {len(incidents)}")

    print("\nTop 10 incidents by risk:")
    print("-" * 100)
    for inc in incidents[:10]:
        print(f"  {inc['incident_id']:<40} "
              f"risk={inc['risk_score']:>5.1f}  "
              f"fanout={inc['max_fanout']}  "
              f"smb={inc['smb_target_count']}  "
              f"logins={inc['total_logins']}")

    summary = summarize(auth_feats, incidents, auth)

    print()
    print("=" * 60)
    print("Auth analytics — summary")
    print("=" * 60)
    for k, v in summary.items():
        print(f"  {k:<24} {v}")

    save_artifacts(auth_feats, incidents, summary)
    return {"auth_feats": auth_feats, "incidents": incidents, "summary": summary}


if __name__ == "__main__":
    print("Loading data ...")
    auth    = pd.read_csv("../data/raw/auth.csv",    parse_dates=["timestamp"])
    netflow = pd.read_csv("../data/raw/netflow.csv", parse_dates=["timestamp"])
    run_auth_analytics(auth, netflow)
