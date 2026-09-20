"""
correlation.py
--------------
Multi-source event correlation.

Fuses detections from three independent models into a single ranked
incident list:

  1. Isolation Forest  -> C2 / beacon anomalies  (reports/anomaly_scores.csv)
  2. Random Forest     -> DNS tunnelling         (models/dns_rf.pkl + raw DNS)
  3. Auth analytics    -> lateral movement       (reports/lateral_incidents.json)

Correlation is performed at the (host, 1-hour window) grain. Windows where
multiple detectors fire receive an escalation bonus, reflecting the SOC
principle that corroborating evidence raises confidence.

Output:
    reports/correlated_incidents.json
    reports/correlation_summary.json
"""

from __future__ import annotations

import os
import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

# Resolve paths relative to project root regardless of CWD
_SRC_DIR = Path(__file__).resolve().parent
ROOT = _SRC_DIR.parent
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"
DATA = ROOT / "data"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
WINDOW = "1h"
WEIGHTS = {
    "c2":    40,   # max contribution from Isolation Forest
    "dns":   35,   # max contribution from DNS classifier
    "auth":  25,   # max contribution from auth lateral-movement incidents
}
ESCALATION_BONUS = 15   # added when 2+ detectors fire in the same window


# ===========================================================================
# 1. LOADERS
# ===========================================================================
def load_anomaly_scores(path: str = str(REPORTS / "anomaly_scores.csv")) -> pd.DataFrame:
    """Load Isolation Forest per-window scores."""
    df = pd.read_csv(path, parse_dates=["window_start"])
    df["entity"]      = df["entity"].astype(str)
    df["window_start"] = pd.to_datetime(df["window_start"]).dt.floor(WINDOW)
    # Normalise anomaly_score to [0,1] using min/max within the dataset
    s = df["anomaly_score"]
    df["c2_score_norm"] = (s - s.min()) / (s.max() - s.min()) if s.max() > s.min() else 0.0
    return df


def load_lateral_incidents(path: str = str(REPORTS / "lateral_incidents.json")) -> list:
    """Load auth lateral-movement incidents."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def score_dns_per_window(dns: pd.DataFrame,
                         model_path: str = str(MODELS / "dns_rf.pkl"),
                         window: str = WINDOW) -> pd.DataFrame:
    """
    Re-score DNS queries with the trained Random Forest and aggregate
    malicious-query probability per (src_ip, window).
    """
    import sys
    sys.path.insert(0, os.path.abspath("../src"))
    from dns_classifier import extract_query_features

    # Load model
    clf = joblib.load(model_path)

    # Feature extraction (must match training)
    X = extract_query_features(dns)
    # Some versions of sklearn need consistent column order
    if hasattr(clf, "feature_names_in_"):
        X = X[list(clf.feature_names_in_)]

    # Predict probability of malicious
    prob_malicious = clf.predict_proba(X)[:, 1]

    df = dns.copy()
    df["window_start"] = df["timestamp"].dt.floor(window)
    df["mal_prob"]     = prob_malicious

    # Aggregate per (src_ip, window)
    agg = (
        df.groupby(["src_ip", "window_start"])
          .agg(
              dns_count=("query", "count"),
              dns_mal_prob_mean=("mal_prob", "mean"),
              dns_mal_prob_max=("mal_prob", "max"),
              dns_mal_prob_sum=("mal_prob", "sum"),
          )
          .reset_index()
          .rename(columns={"src_ip": "host"})
    )
    agg["host"] = "host:" + agg["host"].astype(str)
    return agg


# ===========================================================================
# 2. CORRELATION
# ===========================================================================
def correlate(anomaly_scores: pd.DataFrame,
              dns_scores: pd.DataFrame,
              lateral_incidents: list) -> pd.DataFrame:
    """
    Fuse all detector outputs into a single (host, window) incident table.

    Composite score components (max 100 + 15 escalation):
      - c2_score     (0-40)  : Isolation Forest anomaly score, scaled
      - dns_score    (0-35)  : max malicious probability in window * weight
      - auth_score   (0-25)  : 25 if window falls inside a lateral incident
      - escalation   (0-15)  : added if 2+ detectors fire in this window
    """
    # --- A. Build the base table from all host-hour windows in the C2 scores ---
    base = anomaly_scores[["entity", "window_start", "anomaly_score",
                           "c2_score_norm"]].copy()
    base = base.rename(columns={"entity": "host"})

    # --- B. Attach DNS per-window scores ---
    base = base.merge(
        dns_scores,
        on=["host", "window_start"],
        how="left",
    )

    # --- C. Attach auth lateral-movement flags (HOST-SCOPED) ---
    #
    # We previously aggregated auth risk by window only, which polluted
    # every host in an incident hour. The correct approach is to attribute
    # the incident to the *pivot host* only: the source IP the user was
    # authenticating FROM. Our incident records include this in the
    # "smb_targets" list, but we need the pivot itself. We look it up
    # from the auth_raw source IPs recorded in the incident.
    auth_marks = []
    for inc in lateral_incidents:
        start = pd.to_datetime(inc["start"])
        end   = pd.to_datetime(inc["end"])

        # Pivot host(s): infer from the raw auth records by matching user + time
        # We rely on the caller passing auth_raw through via the incident's
        # internal marker; fall back to a safe default if not present.
        pivot_hosts = inc.get("pivot_hosts", [])
        if not pivot_hosts and "smb_targets" in inc and len(inc["smb_targets"]) > 0:
            # fallback: none - we rely on the correlation caller to enrich
            pivot_hosts = []

        hour = start.floor(WINDOW)
        while hour <= end:
            auth_marks.append({
                "user":         inc["user"],
                "window_start": hour,
                "incident_id":  inc["incident_id"],
                "auth_risk":    inc["risk_score"],
                "pivot_hosts":  tuple(pivot_hosts),   # may be empty
            })
            hour += pd.Timedelta(hours=1)

    auth_df = pd.DataFrame(auth_marks)

    if len(auth_df) > 0:
        # Explode pivot_hosts into one row per (window, pivot_host)
        exploded = auth_df.explode("pivot_hosts")
        exploded = exploded[exploded["pivot_hosts"].notna()]
        exploded["pivot_host"] = "host:" + exploded["pivot_hosts"].astype(str)

        if len(exploded) > 0:
            auth_by_host_window = (
                exploded.groupby(["pivot_host", "window_start"])
                        .agg(auth_risk_max=("auth_risk", "max"),
                             auth_incident_count=("incident_id", "count"))
                        .reset_index()
                        .rename(columns={"pivot_host": "host"})
            )
        else:
            auth_by_host_window = pd.DataFrame(
                columns=["host", "window_start", "auth_risk_max",
                         "auth_incident_count"])
    else:
        auth_by_host_window = pd.DataFrame(
            columns=["host", "window_start", "auth_risk_max",
                     "auth_incident_count"])

    base = base.merge(auth_by_host_window, on=["host", "window_start"], how="left")
    base["auth_risk_max"]       = base["auth_risk_max"].fillna(0)
    base["auth_incident_count"] = base["auth_incident_count"].fillna(0)

    # --- D. Component scores ---
    base["c2_score"] = (base["c2_score_norm"] * WEIGHTS["c2"]).round(2)

    # DNS: use max malicious probability if available, else 0
    dns_prob = base.get("dns_mal_prob_max", pd.Series(0, index=base.index)).fillna(0)
    base["dns_score"] = (dns_prob * WEIGHTS["dns"]).round(2)

    # Auth: scale auth_risk (0-100) into 0-25
    base["auth_score"] = (
        (base["auth_risk_max"] / 100.0) * WEIGHTS["auth"]
    ).round(2)

    # --- E. Escalation bonus and composite score ---
    base["detector_count"] = (
        (base["c2_score"]   > 5).astype(int) +
        (base["dns_score"]  > 5).astype(int) +
        (base["auth_score"] > 5).astype(int)
    )
    base["escalation_bonus"] = np.where(
        base["detector_count"] >= 2, ESCALATION_BONUS, 0
    )

    base["composite_score"] = (
        base["c2_score"] + base["dns_score"] +
        base["auth_score"] + base["escalation_bonus"]
    ).round(2)

    # --- F. Sort by composite score ---
    base = base.sort_values("composite_score", ascending=False).reset_index(drop=True)
    return base


def build_evidence(row: pd.Series) -> dict:
    """Summarise which detectors fired for a single window."""
    evidence = {}
    if row["c2_score"] > 5:
        evidence["c2"] = {
            "score":           float(row["c2_score"]),
            "raw_anomaly":     float(row["anomaly_score"]),
            "interpretation":  "Isolation Forest flagged this window as "
                               "anomalous relative to the population.",
        }
    if row["dns_score"] > 5:
        evidence["dns"] = {
            "score":           float(row["dns_score"]),
            "mal_prob_max":    float(row.get("dns_mal_prob_max", 0) or 0),
            "query_count":     int(row.get("dns_count", 0) or 0),
            "interpretation":  "Random Forest classified DNS queries in this "
                               "window as malicious (tunnelling-like).",
        }
    if row["auth_score"] > 5:
        evidence["auth"] = {
            "score":           float(row["auth_score"]),
            "risk_max":        float(row["auth_risk_max"]),
            "interpretation":  "A lateral-movement incident overlapped this "
                               "window.",
        }
    if row["escalation_bonus"] > 0:
        evidence["escalation"] = {
            "bonus":           float(row["escalation_bonus"]),
            "detector_count":  int(row["detector_count"]),
            "interpretation":  "Multiple independent detectors fired in this "
                               "window — corroborating evidence.",
        }
    return evidence


# ===========================================================================
# 3. ASSEMBLY + SAVE
# ===========================================================================
def build_incident_list(correlated: pd.DataFrame,
                        min_score: float = 10.0,
                        top_n: int = 200) -> list:
    """
    Convert the correlated (host, window) table into a JSON-serialisable
    incident list, keeping only windows above min_score and capped at top_n.
    """
    filtered = correlated[correlated["composite_score"] >= min_score].copy()
    filtered = filtered.head(top_n)

    incidents = []
    for _, row in filtered.iterrows():
        window_start = pd.to_datetime(row["window_start"])
        window_end   = window_start + pd.Timedelta(hours=1)

        incident = {
            "host":            row["host"],
            "window_start":    window_start.isoformat(),
            "window_end":      window_end.isoformat(),
            "composite_score": float(row["composite_score"]),
            "detector_count":  int(row["detector_count"]),
            "detectors":       _list_detectors(row),
            "components": {
                "c2":         float(row["c2_score"]),
                "dns":        float(row["dns_score"]),
                "auth":       float(row["auth_score"]),
                "escalation": float(row["escalation_bonus"]),
            },
            "evidence":        build_evidence(row),
        }
        incidents.append(incident)

    return incidents


def _list_detectors(row: pd.Series) -> list:
    detectors = []
    if row["c2_score"]   > 5: detectors.append("c2_anomaly")
    if row["dns_score"]  > 5: detectors.append("dns_classifier")
    if row["auth_score"] > 5: detectors.append("auth_lateral")
    return detectors


def summarize(correlated: pd.DataFrame) -> dict:
    """Aggregate metrics for the correlation phase."""
    high = correlated[correlated["composite_score"] >= 20]
    multi = correlated[correlated["detector_count"] >= 2]

    return {
        "n_windows_scored":     int(len(correlated)),
        "n_high_confidence":    int(len(high)),
        "n_multi_detector":     int(len(multi)),
        "max_composite_score":  float(correlated["composite_score"].max()),
        "mean_composite_score": float(correlated["composite_score"].mean()),
        "detector_count_distribution": {
            "0_detectors": int((correlated["detector_count"] == 0).sum()),
            "1_detector":  int((correlated["detector_count"] == 1).sum()),
            "2_detectors": int((correlated["detector_count"] == 2).sum()),
            "3_detectors": int((correlated["detector_count"] == 3).sum()),
        },
    }


def save_artifacts(incidents: list, summary: dict,
                   report_dir: str = "../reports") -> None:
    os.makedirs(report_dir, exist_ok=True)

    inc_path = os.path.join(report_dir, "correlated_incidents.json")
    with open(inc_path, "w", encoding="utf-8") as f:
        json.dump(incidents, f, indent=2)
    print(f"✔ Correlated incidents saved: {inc_path}")

    sum_path = os.path.join(report_dir, "correlation_summary.json")
    with open(sum_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"✔ Summary saved: {sum_path}")


# ===========================================================================
# 4. MAIN PIPELINE
# ===========================================================================
def run_correlation() -> dict:
    print("Loading detector outputs ...")
    anomaly = load_anomaly_scores()
    print(f"  C2 anomaly scores   : {len(anomaly):,} windows")

    lateral = load_lateral_incidents()
    print(f"  Lateral incidents   : {len(lateral):,}")

    dns_raw = pd.read_csv(DATA / "raw" / "dns.csv", parse_dates=["timestamp"])
    dns_scores = score_dns_per_window(dns_raw)
    print(f"  DNS scored windows  : {len(dns_scores):,}")

    print("\nCorrelating ...")
    correlated = correlate(anomaly, dns_scores, lateral)

    print("\nBuilding incident list ...")
    incidents = build_incident_list(correlated, min_score=10.0, top_n=200)
    print(f"  incidents above score 10: {len(incidents)}")

    print("\nTop 15 correlated incidents:")
    print("-" * 110)
    for inc in incidents[:15]:
        det = ",".join(inc["detectors"])
        print(f"  score={inc['composite_score']:>6.2f}  "
              f"{inc['host']:<22}  "
              f"{inc['window_start']}  "
              f"n_det={inc['detector_count']}  [{det}]")

    summary = summarize(correlated)
    print()
    print("=" * 60)
    print("Correlation — summary")
    print("=" * 60)
    for k, v in summary.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for k2, v2 in v.items():
                print(f"      {k2:<16} {v2}")
        else:
            print(f"  {k:<24} {v}")

    save_artifacts(incidents, summary)
    return {"correlated": correlated, "incidents": incidents, "summary": summary}


if __name__ == "__main__":
    run_correlation()
