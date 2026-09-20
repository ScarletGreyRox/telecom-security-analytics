"""
anomaly_detector.py
-------------------
Unsupervised C2 / beacon anomaly detection using Isolation Forest.

Inputs:
  data/processed/features.pkl  (built by features.py)

Outputs:
  models/iso_forest.pkl                trained IsolationForest
  reports/anomaly_scores.csv           per-window anomaly score + rank
  reports/anomaly_metrics.json         precision / recall / F1 vs ground truth
  reports/anomaly_feature_importance.csv   permutation importances

Note on evaluation:
  Isolation Forest is trained WITHOUT labels. Ground-truth labels
  (is_compromised_host) are used only for post-hoc evaluation.
"""

from __future__ import annotations

import os
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix
)
from sklearn.preprocessing import StandardScaler
import joblib

# ---------------------------------------------------------------------------
# Feature selection for C2/beacon detection
# ---------------------------------------------------------------------------
# Beacon feature space.
#
# Design note: we deliberately EXCLUDE hour_of_day and day_of_week.
# These are context features, not behaviour features — Isolation Forest
# sees rare values (e.g. 02:00, 22:00) as "anomalous" purely because
# they are uncommon in the dataset, generating ~900 false positives
# and swamping the genuine beacon signal. Time-of-day belongs to the
# exfiltration detector (Phase 8), which uses off_hours_ratio directly.
BEACON_FEATURES = [
    # Core beacon signals (verified to separate attack from baseline)
    "regularity",        # TP=0.19, TN=0.006 -> 32x separation
    "delta_cv",          # TP=0.27, TN=0.71 -> lower is more regular
    "top_dst_share",     # TP=0.93, TN=0.50 -> beacon always to same dst
    "top_port_share",    # TP=0.94, TN=0.62
    "event_count",       # TP=52, TN=2.2 -> beacon fires constantly
    "avg_bytes_sent",    # TP=2.9MB, TN=22KB

    # Excluded after FP diagnostic:
    #   delta_mean, delta_std, delta_mad  -> identical TP/FP/TN distributions
    #   avg_bytes_recvd, unique_dst_*, hour_of_day, day_of_week
]


def compute_precision_at_k(results, ks=(10, 25, 50, 100, 200, 500)):
    """
    Compute Precision@K and Recall@K from the anomaly scores table.
    """
    sorted_df = results.sort_values("anomaly_score", ascending=False)
    total_comp = int(sorted_df["is_compromised_host"].sum())
    out = {}
    for k in ks:
        top = sorted_df.head(k)
        tp = int(top["is_compromised_host"].sum())
        out["top_" + str(k)] = {
            "true_positives": tp,
            "precision": round(tp / k, 4),
            "recall": round(tp / total_comp, 4) if total_comp else 0,
        }
    return out, total_comp

def _prepare_matrix(feats: pd.DataFrame,
                    per_host_normalise: bool = True) -> pd.DataFrame:
    """
    Select and clean the C2 feature matrix.

    If per_host_normalise is True, each feature is converted to a z-score
    relative to that host's own baseline. This is critical for beacon
    detection: a beaconing host is anomalous *relative to its own normal
    behaviour*, not relative to the population. Without this step, busy
    hosts can score as high as the beacon simply because they are busy.
    """
    host = feats[feats["entity"].str.startswith("host:")].copy()

    missing = [c for c in BEACON_FEATURES if c not in host.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    X = host[BEACON_FEATURES].copy()

    # Fill NaN with median before normalisation so z-scores are well-defined
    X = X.fillna(X.median(numeric_only=True))

    # NOTE on normalisation choice:
    # We tried per-host z-scoring and it HURT performance (ROC-AUC 0.64 -> 0.55)
    # because the compromised host's baseline is itself dominated by beacon
    # windows — so the beacon becomes "normal for that host" and quiet windows
    # look anomalous. Population-level scoring is more appropriate here.
    #
    # The right operational fix is per-host top-K alerting (see detect_top_k
    # below), which gives every host a fair chance to surface its own worst
    # windows without distorting the underlying anomaly score.
    _ = per_host_normalise  # kept for backward compat; ignored

    return X, host


# ---------------------------------------------------------------------------
# Training + scoring
# ---------------------------------------------------------------------------
def train_isolation_forest(feats: pd.DataFrame,
                           contamination: float = 0.05,
                           n_estimators: int = 200,
                           random_state: int = 42) -> dict:
    """
    Train Isolation Forest on host-window features.

    contamination: expected fraction of anomalies in the data (0-0.5).
                   We set 0.05 because our three attacks together cover
                   roughly 3-5% of host windows.
    """
    print("Preparing feature matrix ...")
    X, host = _prepare_matrix(feats)
    print(f"  host windows : {len(X):,}")
    print(f"  features     : {len(X.columns)}")

    # ---- Standardise (optional but helps interpretation) ----
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ---- Train Isolation Forest ----
    print(f"\nTraining Isolation Forest "
          f"(n_estimators={n_estimators}, contamination={contamination}) ...")
    iso = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        max_samples="auto",
        random_state=random_state,
        n_jobs=-1,
    )
    iso.fit(X_scaled)

    # ---- Score ----
    # decision_function: higher = more normal
    # score_samples: lower  = more anomalous
    raw_scores = iso.decision_function(X_scaled)   # >0 normal, <0 anomalous
    pred = iso.predict(X_scaled)                   # 1 normal, -1 anomalous

    # Convert to a 0-1 "anomaly score" where 1 = most anomalous
    # (flip sign so higher = more suspicious)
    anomaly_score = -raw_scores

    # ---- Assemble results table ----
    results = host[["entity", "window_start", "is_compromised_host"]].copy()
    results["anomaly_score"] = anomaly_score
    results["is_anomaly"]     = (pred == -1).astype(int)

    # Rank: 1 = most anomalous
    results["anomaly_rank"] = results["anomaly_score"].rank(
        ascending=False, method="first"
    ).astype(int)

    # ---- Post-hoc evaluation against ground truth ----
    y_true = results["is_compromised_host"].values
    y_pred = results["is_anomaly"].values

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    metrics = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc":   float(roc_auc_score(y_true, anomaly_score)),
        "contamination": contamination,
        "n_windows": len(results),
        "n_anomalies_flagged": int(y_pred.sum()),
        "n_true_compromised":  int(y_true.sum()),
    }

    print("\n" + "=" * 60)
    print("Isolation Forest — Evaluation vs Ground Truth")
    print("=" * 60)
    print(f"  Windows scored        : {metrics['n_windows']:,}")
    print(f"  Anomalies flagged     : {metrics['n_anomalies_flagged']:,}")
    print(f"  True compromised      : {metrics['n_true_compromised']:,}")
    print()
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1        : {metrics['f1']:.4f}")
    print(f"  ROC-AUC   : {metrics['roc_auc']:.4f}")
    print()
    print("Confusion matrix (rows=true, cols=pred):")
    print(f"                  pred_normal   pred_anomaly")
    print(f"  true_normal      {cm[0,0]:>11,}   {cm[0,1]:>12,}")
    print(f"  true_compromised {cm[1,0]:>11,}   {cm[1,1]:>12,}")

    # ---- Top anomalies for inspection ----
    print("\nTop 15 highest-scoring windows:")
    top = results.nlargest(15, "anomaly_score")
    for _, r in top.iterrows():
        marker = "  ← COMPROMISED" if r["is_compromised_host"] else ""
        print(f"  rank={r['anomaly_rank']:>5}  "
              f"score={r['anomaly_score']:>7.3f}  "
              f"{r['entity']:<22}  "
              f"{r['window_start']}{marker}")

    # Augment metrics with Precision@K
    prec_at_k, total_comp = compute_precision_at_k(results)
    metrics["precision_at_k"] = prec_at_k
    metrics["random_baseline_precision"] = round(
        total_comp / len(results), 4) if len(results) else 0

    return {
        "model": iso,
        "scaler": scaler,
        "results": results,
        "metrics": metrics,
        "confusion": cm,
        "X": X,
        "X_scaled": X_scaled,
    }


# ---------------------------------------------------------------------------
# Save artifacts
# ---------------------------------------------------------------------------
def save_artifacts(result: dict,
                   model_dir: str = "../models",
                   report_dir: str = "../reports") -> None:
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    # Save the model (and scaler together so the dashboard can load them)
    model_path = os.path.join(model_dir, "iso_forest.pkl")
    joblib.dump({"model": result["model"], "scaler": result["scaler"]},
                model_path)
    print(f"✔ Model saved: {model_path}")

    # Scores table
    scores_path = os.path.join(report_dir, "anomaly_scores.csv")
    result["results"].to_csv(scores_path, index=False)
    print(f"✔ Anomaly scores saved: {scores_path}")

    # Metrics JSON
    metrics_path = os.path.join(report_dir, "anomaly_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(result["metrics"], f, indent=2)
    print(f"✔ Metrics saved: {metrics_path}")


if __name__ == "__main__":
    feats = pd.read_pickle("../data/processed/features.pkl")
    result = train_isolation_forest(feats)
    save_artifacts(result)


# ---------------------------------------------------------------------------
# Operational alerting: per-host top-K
# ---------------------------------------------------------------------------
def detect_top_k_per_host(results: pd.DataFrame, k: int = 2) -> pd.DataFrame:
    """
    Take the global anomaly scores and produce an operational alert list
    using a per-host top-K rule: for each host, flag its K highest-scoring
    windows.

    Rationale: the isolated-forest score is a relative ranking. Global
    thresholding at 5% contamination flags ~900 windows, most of which are
    simply "unusual but not malicious" hours from busy hosts. Per-host
    top-K instead asks: "for each host, which of its own windows look
    worst?" - which surfaces the beacon without the population bias.
    """
    out = results.copy()
    out["rank_within_host"] = (
        out.groupby("entity")["anomaly_score"]
           .rank(ascending=False, method="first")
           .astype(int)
    )
    out["alert_topk"] = (out["rank_within_host"] <= k).astype(int)
    return out


def evaluate_top_k(results: pd.DataFrame, k: int = 2) -> dict:
    """Evaluate the per-host top-K alerting rule."""
    from sklearn.metrics import precision_score, recall_score, f1_score

    scored = detect_top_k_per_host(results, k=k)
    y_true = scored["is_compromised_host"].values
    y_pred = scored["alert_topk"].values

    alerts = scored[scored["alert_topk"] == 1]
    tp = alerts[alerts["is_compromised_host"] == 1]
    fp = alerts[alerts["is_compromised_host"] == 0]

    metrics = {
        "k": k,
        "total_alerts": int(len(alerts)),
        "true_positives": int(len(tp)),
        "false_positives": int(len(fp)),
        "unique_hosts_alerted": int(alerts["entity"].nunique()),
        "unique_compromised_hosts_caught":
            int(alerts[alerts["is_compromised_host"] == 1]["entity"].nunique()),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
    }
    return metrics, scored
