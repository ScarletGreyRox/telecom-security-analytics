"""
mitre_mapping.py
----------------
Map correlated incidents to MITRE ATT&CK techniques.

Design: static, detector-driven mapping (Option A).
Each detector contributes a fixed set of techniques with confidence
levels. An incident's technique list is the union of the mappings of
all detectors that fired for it.

Outputs:
    reports/mitre_tagged_incidents.json
    reports/mitre_summary.json
"""

from __future__ import annotations

import os
import json
from collections import Counter
from pathlib import Path

# Resolve paths relative to project root regardless of CWD
_SRC_DIR = Path(__file__).resolve().parent
ROOT = _SRC_DIR.parent
REPORTS = ROOT / "reports"

# ---------------------------------------------------------------------------
# Static technique catalogue (subset of MITRE ATT&CK Enterprise)
# ---------------------------------------------------------------------------
TECHNIQUES = {
    "T1071":     {"name": "Application Layer Protocol",
                  "tactic": "Command and Control"},
    "T1071.004": {"name": "Application Layer Protocol: DNS",
                  "tactic": "Command and Control"},
    "T1568":     {"name": "Dynamic Resolution",
                  "tactic": "Command and Control"},
    "T1568.002": {"name": "Domain Generation Algorithms",
                  "tactic": "Command and Control"},
    "T1571":     {"name": "Non-Standard Port",
                  "tactic": "Command and Control"},
    "T1105":     {"name": "Ingress Tool Transfer",
                  "tactic": "Command and Control"},
    "T1041":     {"name": "Exfiltration Over C2 Channel",
                  "tactic": "Exfiltration"},
    "T1048":     {"name": "Exfiltration Over Alternative Protocol",
                  "tactic": "Exfiltration"},
    "T1048.003": {"name": "Exfiltration Over Alternative Protocol: DNS",
                  "tactic": "Exfiltration"},
    "T1021.001": {"name": "Remote Services: RDP",
                  "tactic": "Lateral Movement"},
    "T1021.002": {"name": "Remote Services: SMB/Windows Admin Shares",
                  "tactic": "Lateral Movement"},
    "T1078":     {"name": "Valid Accounts",
                  "tactic": "Defense Evasion, Persistence"},
    "T1110":     {"name": "Brute Force",
                  "tactic": "Credential Access"},
}


# ---------------------------------------------------------------------------
# Static detector -> techniques mapping
# ---------------------------------------------------------------------------
DETECTOR_TECHNIQUES = {
    "c2_anomaly": [
        ("T1071", "high"),
        ("T1568", "medium"),
        ("T1571", "low"),
        ("T1041", "medium"),
    ],
    "dns_classifier": [
        ("T1071.004", "high"),
        ("T1048.003", "medium"),
        ("T1568.002", "low"),
    ],
    "auth_lateral": [
        ("T1021.002", "high"),
        ("T1078",     "high"),
        ("T1021.001", "medium"),
    ],
}


def map_incident_to_techniques(incident: dict) -> list:
    """
    Static detector-driven mapping.

    Returns a list of technique dicts (id, name, tactic, confidence, source).
    Techniques from multiple detectors are unioned; duplicates removed.
    """
    tagged = []
    seen = set()

    for det in incident.get("detectors", []):
        for tech_id, conf in DETECTOR_TECHNIQUES.get(det, []):
            if tech_id in seen:
                continue
            seen.add(tech_id)
            tagged.append({
                "id":         tech_id,
                "name":       TECHNIQUES[tech_id]["name"],
                "tactic":     TECHNIQUES[tech_id]["tactic"],
                "confidence": conf,
                "source":     f"detector:{det}",
            })

    return tagged


# ===========================================================================
# PIPELINE
# ===========================================================================
def load_enriched(path: str = str(REPORTS / "enriched_incidents.json")) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def tag_incidents(incidents: list) -> list:
    """Apply MITRE mapping to every incident."""
    tagged = []
    for inc in incidents:
        techniques = map_incident_to_techniques(inc)
        new_inc = dict(inc)
        new_inc["mitre_techniques"] = techniques
        new_inc["mitre_tactics"]    = sorted({t["tactic"] for t in techniques})
        new_inc["technique_count"]  = len(techniques)
        tagged.append(new_inc)
    return tagged


def summarize(tagged: list) -> dict:
    """Aggregate technique and tactic coverage across incidents."""
    technique_counts = Counter()
    tactic_counts    = Counter()
    tactic_pair_counts = Counter()

    for inc in tagged:
        seen_tech = set()
        seen_tact = set()
        for t in inc["mitre_techniques"]:
            if t["id"] not in seen_tech:
                technique_counts[t["id"]] += 1
                seen_tech.add(t["id"])
            if t["tactic"] not in seen_tact:
                tactic_counts[t["tactic"]] += 1
                seen_tact.add(t["tactic"])
        # distinct tactic-pair per incident
        pair = tuple(sorted(seen_tact))
        if pair:
            tactic_pair_counts[pair] += 1

    return {
        "n_incidents":              len(tagged),
        "unique_techniques_seen":   len(technique_counts),
        "unique_tactics_seen":      len(tactic_counts),
        "technique_frequency":      dict(technique_counts.most_common()),
        "tactic_frequency":         dict(tactic_counts.most_common()),
        "tactic_pair_frequency":    {
            " + ".join(k): v for k, v in tactic_pair_counts.most_common(10)
        },
        "mean_techniques_per_incident":
            round(sum(inc["technique_count"] for inc in tagged) / len(tagged), 2)
            if tagged else 0,
    }


def save_artifacts(tagged: list, summary: dict,
                   report_dir: str = str(REPORTS)) -> None:
    os.makedirs(report_dir, exist_ok=True)

    inc_path = os.path.join(report_dir, "mitre_tagged_incidents.json")
    with open(inc_path, "w", encoding="utf-8") as f:
        json.dump(tagged, f, indent=2)
    print(f"✔ Tagged incidents saved: {inc_path}")

    sum_path = os.path.join(report_dir, "mitre_summary.json")
    with open(sum_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"✔ Summary saved: {sum_path}")


def run_mitre_mapping() -> dict:
    print("Loading enriched incidents ...")
    incidents = load_enriched()
    print(f"  incidents: {len(incidents)}")

    print("\nApplying MITRE ATT&CK mapping ...")
    tagged = tag_incidents(incidents)

    print("\nTop 10 incidents with technique coverage:")
    print("-" * 120)
    for inc in tagged[:10]:
        techs = ", ".join(t["id"] for t in inc["mitre_techniques"])
        print(f"  {inc['host']:<22} {inc['window_start']}  "
              f"final={inc['threat_intel']['final_score']:>6.2f}  "
              f"tactics={len(inc['mitre_tactics'])}  "
              f"techniques=[{techs}]")

    summary = summarize(tagged)
    print()
    print("=" * 60)
    print("MITRE ATT&CK mapping — summary")
    print("=" * 60)
    print(f"  Incidents                  : {summary['n_incidents']}")
    print(f"  Unique techniques observed : {summary['unique_techniques_seen']}")
    print(f"  Unique tactics observed    : {summary['unique_tactics_seen']}")
    print(f"  Mean techniques/incident   : "
          f"{summary['mean_techniques_per_incident']}")
    print()
    print("  Technique frequency:")
    for tid, count in summary["technique_frequency"].items():
        name = TECHNIQUES.get(tid, {}).get("name", "?")
        print(f"    {tid:<12} {count:>4} incidents  {name}")
    print()
    print("  Tactic frequency:")
    for tactic, count in summary["tactic_frequency"].items():
        print(f"    {tactic:<30} {count:>4} incidents")

    save_artifacts(tagged, summary)
    return {"tagged": tagged, "summary": summary}


if __name__ == "__main__":
    run_mitre_mapping()
