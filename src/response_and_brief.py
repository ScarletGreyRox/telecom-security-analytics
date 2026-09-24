"""response_and_brief.py — generates response playbook and executive brief."""
from __future__ import annotations

import os
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent
ROOT = _SRC_DIR.parent
REPORTS = ROOT / "reports"


def load_incidents():
    p = REPORTS / "mitre_tagged_incidents.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p} - run the main pipeline first")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def load_summary(name):
    p = REPORTS / name
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


PLAYBOOK = {
    "T1071": {"name": "Application Layer Protocol",
              "containment": ["Block outbound C2 at perimeter", "Sinkhole C2 domains", "Isolate affected host"],
              "eradication": ["Image-rebuild host", "Rotate cached credentials"],
              "recovery": ["Restore after 48h clean monitoring"]},
    "T1071.004": {"name": "Application Layer Protocol: DNS",
                  "containment": ["Sinkhole tunnelling domain", "Rate-limit DNS queries"],
                  "eradication": ["Remove tunnelling client", "Scan adjacent hosts"],
                  "recovery": ["Restore DNS after verifying clean"]},
    "T1568": {"name": "Dynamic Resolution",
              "containment": ["Enable DNSSEC validation", "Block unclassified TLDs"],
              "eradication": ["Remove fallback-channel config"],
              "recovery": ["Monitor for re-emergence"]},
    "T1568.002": {"name": "Domain Generation Algorithms",
                  "containment": ["Enable DGA detection on resolvers"],
                  "eradication": ["Remove DGA client components"],
                  "recovery": ["Monitor DNS entropy for 14 days"]},
    "T1571": {"name": "Non-Standard Port",
              "containment": ["Restrict outbound to approved ports"],
              "eradication": ["Remove unauthorised listener services"],
              "recovery": ["Re-enable ports after review"]},
    "T1041": {"name": "Exfiltration Over C2 Channel",
              "containment": ["Throttle outbound to 1 Mbps", "Block exfil destinations"],
              "eradication": ["Remove staging data", "Rotate leaked credentials"],
              "recovery": ["Restore bandwidth after forensics"]},
    "T1048": {"name": "Exfiltration Over Alternative Protocol",
              "containment": ["Block external TXT queries", "Enable DPI"],
              "eradication": ["Remove alternative-protocol clients"],
              "recovery": ["Resume for approved business use only"]},
    "T1048.003": {"name": "Exfiltration Over Alternative Protocol: DNS",
                  "containment": ["Sinkhole DNS entropy > 4.0"],
                  "eradication": ["Remove DNS-tunnel client"],
                  "recovery": ["Resume external DNS after verifying clean"]},
    "T1021.001": {"name": "Remote Services: RDP",
                  "containment": ["Disable RDP on source host", "Block source IP"],
                  "eradication": ["Audit RDP sessions for 72h"],
                  "recovery": ["Re-enable with MFA enforced"]},
    "T1021.002": {"name": "Remote Services: SMB",
                  "containment": ["Disable SMBv1", "Revoke admin share access"],
                  "eradication": ["Remove scheduled tasks", "Sweep pivot hosts"],
                  "recovery": ["Restore SMB after verifying clean"]},
    "T1078": {"name": "Valid Accounts",
              "containment": ["Disable user account in AD", "Force password reset"],
              "eradication": ["Rotate credentials user accessed", "Review MFA"],
              "recovery": ["Re-enable account after clean log"]},
    "T1110": {"name": "Brute Force",
              "containment": ["Enable lockout after 5 failures", "Block source IP"],
              "eradication": ["Force password reset"],
              "recovery": ["Monitor auth logs for re-emergence"]},
    "T1105": {"name": "Ingress Tool Transfer",
              "containment": ["Block unclassified file-sharing domains"],
              "eradication": ["Remove transferred tooling"],
              "recovery": ["Re-scan host after 24h"]},
}


def build_playbook(incidents):
    plans = []
    for inc in incidents[:50]:
        techniques = [t["id"] for t in inc.get("mitre_techniques", [])]
        containment, eradication, recovery = [], [], []
        for t in techniques:
            entry = PLAYBOOK.get(t)
            if not entry:
                continue
            containment.extend(entry["containment"])
            eradication.extend(entry["eradication"])
            recovery.extend(entry["recovery"])
        containment = list(dict.fromkeys(containment))
        eradication = list(dict.fromkeys(eradication))
        recovery = list(dict.fromkeys(recovery))
        plans.append({
            "incident_id": inc.get("host", "") + "@" + inc.get("window_start", ""),
            "host": inc.get("host", ""),
            "window_start": inc.get("window_start", ""),
            "final_score": inc.get("threat_intel", {}).get("final_score", 0),
            "techniques": techniques,
            "containment": containment,
            "eradication": eradication,
            "recovery": recovery,
        })
    return plans


def build_brief(incidents, mitre_summary, ti_summary):
    n_total = len(incidents)
    high_risk = [i for i in incidents
                 if i.get("threat_intel", {}).get("final_score", 0) >= 100]
    actors = ti_summary.get("threat_actor_distribution", {})
    tactics = mitre_summary.get("tactic_frequency", {})

    top = incidents[0] if incidents else {}
    top_host = top.get("host", "N/A").replace("host:", "")
    top_score = top.get("threat_intel", {}).get("final_score", 0)
    top_tech_count = len(top.get("mitre_techniques", []))

    lines = [
        "# Executive Security Brief",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Period:** 2026-08-10 to 2026-08-16 (7-day synthetic capture)",
        "",
        "## Headline",
        "",
        f"- **{n_total} correlated incidents** in the queue",
        f"- **{len(high_risk)} high-confidence** incidents",
        f"- **{len(actors)} threat actors** attributed",
        f"- **{len(tactics)} MITRE ATT&CK tactics** observed",
        "",
        "## Top incident",
        "",
        f"Host **{top_host}** produced the highest-scoring incident (final score "
        f"**{top_score}**), spanning **{top_tech_count} ATT&CK techniques**.",
        "",
        "## Threat actor landscape",
        "",
    ]
    for actor, count in actors.items():
        lines.append(f"- **{actor}**: {count} incidents matched")
    lines.append("")
    lines.append("## MITRE ATT&CK coverage")
    lines.append("")
    for tactic, count in tactics.items():
        lines.append(f"- **{tactic}**: {count} incidents")
    lines.append("")
    lines.append("## Recommended actions")
    lines.append("")
    lines.append("1. Isolate the top-ranked compromised host")
    lines.append("2. Disable the pivot user account in Active Directory")
    lines.append("3. Block external IOC infrastructure at perimeter")
    lines.append("4. Rotate credentials accessed by the pivot user")
    lines.append("5. Verify SMB traffic baseline after remediation")
    lines.append("")
    return "\n".join(lines)


def run_all():
    print("Loading incidents ...")
    incidents = load_incidents()
    print(f"  {len(incidents)} incidents")

    mitre_summary = load_summary("mitre_summary.json")
    ti_summary = load_summary("threat_intelligence_summary.json")

    print("Building response playbook ...")
    plans = build_playbook(incidents)
    playbook_out = REPORTS / "response_playbook.json"
    with open(playbook_out, "w", encoding="utf-8") as f:
        json.dump({"n_plans": len(plans), "plans": plans}, f, indent=2)
    print(f"  OK: {playbook_out}")

    print("Building executive brief ...")
    brief = build_brief(incidents, mitre_summary, ti_summary)
    brief_out = REPORTS / "executive_brief.md"
    with open(brief_out, "w", encoding="utf-8") as f:
        f.write(brief)
    print(f"  OK: {brief_out}")
    print("Done.")


if __name__ == "__main__":
    run_all()
