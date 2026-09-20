"""
dns_classifier.py
-----------------
Random Forest classifier that distinguishes benign from malicious DNS queries.

Training data: DNS query logs from data/raw/dns.csv
Labels:        query labelled malicious if its base domain is in the known
               malicious-domain list from data_generator.py

Outputs:
  - models/dns_rf.pkl             trained RandomForestClassifier
  - reports/dns_confusion.csv     confusion matrix
  - reports/dns_metrics.json      precision / recall / F1
  - reports/dns_feature_importance.csv
"""

from __future__ import annotations

import os
import json
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_score, recall_score, f1_score,
    roc_auc_score
)
import joblib

# ---------------------------------------------------------------------------
# Malicious domains used by the synthetic generator
# ---------------------------------------------------------------------------
MALICIOUS_DOMAINS = {
    "evil-c2.example",
    "badactor-update.net",
    "dns-tunnel.xyz",
    "malware-cdn.top",
    "apt-relay.cc",
}


# ---------------------------------------------------------------------------
# Query-level feature extraction
# ---------------------------------------------------------------------------
def _entropy(s: str) -> float:
    """Shannon entropy in bits per character."""
    if not s:
        return 0.0
    counts = Counter(s)
    n = len(s)
    return -sum((c / n) * np.log2(c / n) for c in counts.values())


def extract_query_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build per-query features from raw DNS logs.

    The query is split into subdomain (first label) and base domain
    (everything after the first label).
    """
    q = df["query"].astype(str)
    parts = q.str.split(".", n=1)

    subdomain   = parts.str[0].fillna("")
    base_domain = parts.str[1].fillna("")

    # IMPORTANT: we deliberately EXCLUDE base-domain features
    # (base_len, base_entropy, num_labels) because they encode the
    # identity of the malicious base domain itself, rather than the
    # *behaviour* of the subdomain. Including them causes the model to
    # memorise specific malicious domains instead of learning
    # behavioural signatures of tunnelling, which would fail against
    # unseen domains in production.
    feats = pd.DataFrame({
        "sub_len":       subdomain.str.len(),
        "sub_entropy":   subdomain.apply(_entropy),
        "sub_digit_ratio":
            subdomain.str.count(r"\d") /
            subdomain.str.len().replace(0, np.nan),
        "sub_hyphen_count": subdomain.str.count("-"),
        "is_long_query": (q.str.len() > 30).astype(int),
        "query_type_txt": (df["query_type"] == "TXT").astype(int),
        "query_type_aaaa": (df["query_type"] == "AAAA").astype(int),
        "is_nxdomain":   (df["response_code"] == "NXDOMAIN").astype(int),
    })

    return feats


def build_labelled_dns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract features and ground-truth labels from raw DNS logs.

    A query is labelled malicious if its base domain is in MALICIOUS_DOMAINS.
    """
    X = extract_query_features(df)
    y = df["domain"].isin(MALICIOUS_DOMAINS).astype(int)
    return X, y


# ---------------------------------------------------------------------------
# Training + evaluation
# ---------------------------------------------------------------------------
def train_dns_classifier(df: pd.DataFrame,
                         test_size: float = 0.25,
                         random_state: int = 42,
                         n_estimators: int = 200) -> dict:
    """
    Train a Random Forest on labelled DNS queries and evaluate it.

    Returns a dict with the model, metrics, and feature importances.
    """
    print("Extracting features and labels ...")
    X, y = build_labelled_dns(df)
    print(f"  total queries : {len(X):,}")
    print(f"  malicious     : {int(y.sum()):,} ({100*y.mean():.2f}%)")
    print(f"  benign        : {int((1-y).sum()):,}")

    # --- Train/test split, stratified to preserve class ratio ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"\nTrain: {len(X_train):,}  |  Test: {len(X_test):,}")

    # --- Train Random Forest ---
    print("\nTraining Random Forest ...")
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",   # corrects for ~4% positive rate
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    # --- Predict on test set ---
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    # --- Metrics ---
    metrics = {
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_test, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc":   float(roc_auc_score(y_test, y_prob)),
        "n_train":   int(len(X_train)),
        "n_test":    int(len(X_test)),
        "n_positive": int(y.sum()),
    }

    print("\n" + "=" * 60)
    print("Random Forest DNS Classifier — Evaluation")
    print("=" * 60)
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1        : {metrics['f1']:.4f}")
    print(f"  ROC-AUC   : {metrics['roc_auc']:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion matrix (rows=true, cols=pred):")
    print(f"                pred_benign   pred_malicious")
    print(f"  true_benign   {cm[0,0]:>10,}   {cm[0,1]:>13,}")
    print(f"  true_mal     {cm[1,0]:>10,}   {cm[1,1]:>13,}")

    print("\nClassification report:")
    print(classification_report(y_test, y_pred,
                                target_names=["benign", "malicious"],
                                zero_division=0))

    # --- Feature importances ---
    importances = pd.DataFrame({
        "feature":    X.columns,
        "importance": clf.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    print("Feature importances (top 11):")
    for _, r in importances.iterrows():
        bar = "█" * int(round(r["importance"] * 50))
        print(f"  {r['feature']:<20} {r['importance']:.4f}  {bar}")

    return {
        "model": clf,
        "metrics": metrics,
        "confusion": cm,
        "feature_importances": importances,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


def save_artifacts(result: dict,
                   model_dir: str = "../models",
                   report_dir: str = "../reports") -> None:
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    # Model
    model_path = os.path.join(model_dir, "dns_rf.pkl")
    joblib.dump(result["model"], model_path)
    print(f"✔ Model saved: {model_path}")

    # Metrics JSON
    metrics_path = os.path.join(report_dir, "dns_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(result["metrics"], f, indent=2)
    print(f"✔ Metrics saved: {metrics_path}")

    # Confusion matrix CSV
    cm_path = os.path.join(report_dir, "dns_confusion.csv")
    pd.DataFrame(
        result["confusion"],
        index=["true_benign", "true_malicious"],
        columns=["pred_benign", "pred_malicious"],
    ).to_csv(cm_path)
    print(f"✔ Confusion matrix saved: {cm_path}")

    # Feature importances CSV
    fi_path = os.path.join(report_dir, "dns_feature_importance.csv")
    result["feature_importances"].to_csv(fi_path, index=False)
    print(f"✔ Feature importances saved: {fi_path}")


if __name__ == "__main__":
    dns = pd.read_csv("../data/raw/dns.csv", parse_dates=["timestamp"])
    result = train_dns_classifier(dns)
    save_artifacts(result)
