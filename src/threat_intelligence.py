"""
threat_intelligence.py
----------------------
Threat intelligence enrichment for correlated incidents.

Two enrichment mechanisms:
  1. IOC matching    - exact-match IPs and domains against a known-bad feed
  2. Pattern similarity - TF-IDF + cosine similarity against a library of
                          known attack patterns (per charter requirements)

Outputs:
    reports/enriched_incidents.json
    reports/threat_intelligence_summary.json
"""

from __future__ import annotations

import os
import json
import re
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Resolve paths relative to project root regardless of CWD
_SRC_DIR = Path(__file__).resolve().parent
ROOT = _SRC_DIR.parent
REPORTS = ROOT / "reports"
DATA = ROOT / "data"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
IOC_PATH  = str(DATA / "threat_intelligence" / "ioc_list.csv")
PAT_PATH  = str(DATA / "threat_intelligence" / "attack_patterns.csv")
CORR_PATH = str(REPORTS / "correlated_incidents.json")
SMB_PORTS = {445, 3389}
TI_MAX_BONUS = 15.0     # max points added to composite score from TI


# ===========================================================================
# 1. LOADERS
# ===========================================================================
def load_iocs(path: str = IOC_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["value_lower"] = df["value"].str.lower()
    return df


def load_patterns(path: str = PAT_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def load_incidents(path: str = CORR_PATH) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ===========================================================================
# 2. IOC MATCHING
# ===========================================================================
def match_iocs(incident: dict,
               iocs: pd.DataFrame,
               netflow: pd.DataFrame,
               dns: pd.DataFrame) -> list:
    """
    Find IOC matches relevant to an incident.

    Strategy:
      - The incident's host and window tell us *when* and *where*.
      - We look at all netflow destinations and all DNS queries from that
        host during the window, and check them against the IOC feed.
      - The incident itself doesn't carry raw IPs / domains, so we look them
        up from the raw telemetry (this is how a real SOC enrichment works).
    """
    host_raw = incident["host"].replace("host:", "")
    win_start = pd.to_datetime(incident["window_start"])
    win_end   = pd.to_datetime(incident["window_end"])

    # --- netflow destinations from this host in the window ---
    nf = netflow[
        (netflow["src_ip"] == host_raw) &
        (netflow["timestamp"] >= win_start) &
        (netflow["timestamp"] <  win_end)
    ]
    dest_ips = set(nf["dst_ip"].astype(str).unique().tolist())

    # --- DNS base domains from this host in the window ---
    dns_w = dns[
        (dns["src_ip"] == host_raw) &
        (dns["timestamp"] >= win_start) &
        (dns["timestamp"] <  win_end)
    ]
    base_domains = set()
    for q in dns_w["query"].astype(str):
        parts = q.split(".", 1)
        if len(parts) == 2:
            base_domains.add(parts[1].lower())
        else:
            base_domains.add(q.lower())

    # --- Match against the IOC feed ---
    matches = []
    for _, ioc in iocs.iterrows():
        ioc_type = ioc["ioc_type"]
        ioc_val  = ioc["value_lower"]

        if ioc_type == "ip" and ioc_val in {ip.lower() for ip in dest_ips}:
            matches.append({
                "ioc_type":     "ip",
                "value":        ioc["value"],
                "threat_actor": ioc["threat_actor"],
                "confidence":   float(ioc["confidence"]),
                "description":  ioc["description"],
            })
        elif ioc_type == "domain" and ioc_val in base_domains:
            matches.append({
                "ioc_type":     "domain",
                "value":        ioc["value"],
                "threat_actor": ioc["threat_actor"],
                "confidence":   float(ioc["confidence"]),
                "description":  ioc["description"],
            })

    return matches


# ===========================================================================
# 3. INCIDENT TEXT + TF-IDF PATTERN SIMILARITY
# ===========================================================================
def incident_to_text(incident: dict) -> str:
    """
    Convert an incident into a natural-language-ish string that captures
    its behavioural signature. This text is what we compare against the
    known-attack pattern library using TF-IDF.
    """
    parts = []

    # Which detectors fired?
    for det in incident.get("detectors", []):
        if det == "c2_anomaly":
            parts.append("anomaly detection beaconing periodic timing regularity")
        elif det == "dns_classifier":
            parts.append("dns query entropy subdomain unusual malicious tunnel")
        elif det == "auth_lateral":
            parts.append("authentication fanout lateral movement credential pivot")

    # Score components
    comp = incident.get("components", {})
    if comp.get("c2", 0)   > 5: parts.append("c2 beacon anomaly score high")
    if comp.get("dns", 0)  > 5: parts.append("dns suspicious queries")
    if comp.get("auth", 0) > 5: parts.append("auth risk lateral")
    if comp.get("escalation", 0) > 0:
        parts.append("multi detector corroboration escalation")

    # Host / window context
    host = incident.get("host", "")
    if "10.0.3" in host:
        parts.append("subnet 10.0.3")
    if "10.0.2" in host:
        parts.append("subnet 10.0.2")

    win_start = incident.get("window_start", "")
    hour = -1
    try:
        hour = int(win_start[11:13])
    except Exception:
        pass
    if 0 <= hour < 6:
        parts.append("off hours night transfer exfiltration")
    elif 6 <= hour < 12:
        parts.append("morning hours")
    elif 12 <= hour < 18:
        parts.append("afternoon business hours")
    else:
        parts.append("evening hours")

    # Evidence interpretation strings
    ev = incident.get("evidence", {})
    if "dns" in ev:
        parts.append(ev["dns"].get("interpretation", ""))
    if "auth" in ev:
        parts.append(ev["auth"].get("interpretation", ""))

    return " ".join(p for p in parts if p).lower()


def build_pattern_similarity(incidents: list,
                              patterns: pd.DataFrame) -> tuple:
    """
    Fit TF-IDF over the combined corpus of incident texts + pattern
    descriptions, then compute cosine similarity between each incident
    and each pattern.

    Returns:
      (similarity_matrix, incident_texts, pattern_texts, vectorizer)
    """
    incident_texts = [incident_to_text(inc) for inc in incidents]
    pattern_texts  = list(patterns["description"].astype(str))

    corpus = incident_texts + pattern_texts

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
    )
    tfidf = vectorizer.fit_transform(corpus)

    inc_mat = tfidf[: len(incident_texts)]
    pat_mat = tfidf[len(incident_texts):]

    sim = cosine_similarity(inc_mat, pat_mat)
    return sim, incident_texts, pattern_texts, vectorizer


def top_similar_patterns(sim_row: np.ndarray,
                          patterns: pd.DataFrame,
                          k: int = 3,
                          min_similarity: float = 0.05) -> list:
    """Return the k most similar patterns for a single incident."""
    order = np.argsort(-sim_row)
    result = []
    for idx in order[:k]:
        score = float(sim_row[idx])
        if score < min_similarity:
            continue
        result.append({
            "pattern_id":  patterns.iloc[idx]["pattern_id"],
            "name":        patterns.iloc[idx]["name"],
            "description": patterns.iloc[idx]["description"],
            "similarity":  round(score, 4),
        })
    return result


# ===========================================================================
# 4. TI SCORING + ENRICHMENT
# ===========================================================================
def compute_ti_score(matches: list, similar: list) -> float:
    """
    Compute the TI bonus (0 to TI_MAX_BONUS).

    Components:
      - IOC matches     : up to 12 pts (sum of confidence / max_iocs, capped)
      - Pattern sim     : up to 3 pts  (max similarity, scaled)
    """
    # IOC component (0-12)
    if matches:
        # Sum of confidence scores of top 3 matches, capped at 1.0
        conf_sum = sum(m["confidence"] for m in matches[:3])
        ioc_component = min(conf_sum, 1.0) * 12.0
    else:
        ioc_component = 0.0

    # Pattern similarity component (0-3)
    if similar:
        max_sim = max(s["similarity"] for s in similar)
        pattern_component = min(max_sim, 1.0) * 3.0
    else:
        pattern_component = 0.0

    return round(min(ioc_component + pattern_component, TI_MAX_BONUS), 2)


def enrich_incidents(incidents: list,
                     iocs: pd.DataFrame,
                     patterns: pd.DataFrame,
                     netflow: pd.DataFrame,
                     dns: pd.DataFrame) -> list:
    """
    Enrich every incident with IOC matches + similar known attacks + TI score.
    """
    # Precompute TF-IDF similarity matrix in one pass
    print("  Building TF-IDF similarity matrix ...")
    sim, _, _, _ = build_pattern_similarity(incidents, patterns)

    enriched = []
    print(f"  Enriching {len(incidents)} incidents ...")
    for i, inc in enumerate(incidents):
        matches = match_iocs(inc, iocs, netflow, dns)
        similar = top_similar_patterns(sim[i], patterns, k=3)
        ti_score = compute_ti_score(matches, similar)

        new_inc = dict(inc)   # copy
        new_inc["threat_intel"] = {
            "matched_iocs":     matches,
            "similar_attacks":  similar,
            "ti_score":         ti_score,
            "final_score":      round(inc["composite_score"] + ti_score, 2),
            "matched_ioc_count": len(matches),
            "threat_actors":    sorted({m["threat_actor"] for m in matches}),
        }
        enriched.append(new_inc)

    # Re-sort by final score
    enriched.sort(key=lambda x: -x["threat_intel"]["final_score"])
    return enriched


def summarize(enriched: list) -> dict:
    """Aggregate metrics for the report."""
    n = len(enriched)
    with_ioc  = sum(1 for i in enriched if i["threat_intel"]["matched_ioc_count"] > 0)
    actors    = {}
    for i in enriched:
        for a in i["threat_intel"]["threat_actors"]:
            actors[a] = actors.get(a, 0) + 1

    return {
        "n_incidents":            n,
        "incidents_with_ioc_match": with_ioc,
        "ioc_match_rate":         round(with_ioc / n, 4) if n else 0,
        "threat_actor_distribution": actors,
        "max_final_score":        max((i["threat_intel"]["final_score"]
                                       for i in enriched), default=0),
        "mean_final_score":       round(
            sum(i["threat_intel"]["final_score"] for i in enriched) / n, 2
        ) if n else 0,
    }


def save_artifacts(enriched: list, summary: dict,
                   report_dir: str = str(REPORTS)) -> None:
    os.makedirs(report_dir, exist_ok=True)

    out_path = os.path.join(report_dir, "enriched_incidents.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(enriched, f, indent=2)
    print(f"✔ Enriched incidents saved: {out_path}")

    sum_path = os.path.join(report_dir, "threat_intelligence_summary.json")
    with open(sum_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"✔ Summary saved: {sum_path}")


# ===========================================================================
# 5. MAIN PIPELINE
# ===========================================================================
def run_threat_intelligence() -> dict:
    print("Loading threat intelligence and correlated incidents ...")
    iocs     = load_iocs()
    patterns = load_patterns()
    incidents = load_incidents()
    print(f"  IOCs          : {len(iocs)}")
    print(f"  Patterns      : {len(patterns)}")
    print(f"  Incidents     : {len(incidents)}")

    print("\nLoading raw telemetry for enrichment lookup ...")
    netflow = pd.read_csv(DATA / "raw" / "netflow.csv", parse_dates=["timestamp"])
    dns     = pd.read_csv(DATA / "raw" / "dns.csv",     parse_dates=["timestamp"])

    print("\nEnriching incidents ...")
    enriched = enrich_incidents(incidents, iocs, patterns, netflow, dns)

    print("\nTop 10 enriched incidents:")
    print("-" * 120)
    for inc in enriched[:10]:
        ti = inc["threat_intel"]
        ioc_str = ", ".join(f"{m['ioc_type']}={m['value']}"
                            for m in ti["matched_iocs"][:3]) or "-"
        print(f"  {inc['host']:<22} {inc['window_start']}  "
              f"composite={inc['composite_score']:>6.2f}  "
              f"ti=+{ti['ti_score']:>5.2f}  "
              f"final={ti['final_score']:>6.2f}  "
              f"iocs=[{ioc_str}]")

    summary = summarize(enriched)
    print()
    print("=" * 60)
    print("Threat intelligence — summary")
    print("=" * 60)
    for k, v in summary.items():
        print(f"  {k:<28} {v}")

    save_artifacts(enriched, summary)
    return {"enriched": enriched, "summary": summary}


if __name__ == "__main__":
    run_threat_intelligence()
