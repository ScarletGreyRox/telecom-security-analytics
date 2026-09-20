"""
app.py
------
Streamlit SOC dashboard for the SAS821S Capstone Project.

Loads all reports produced by the detection pipeline and presents them
in a SOC-analyst workflow:

    Overview | Incident Queue | Incident Detail | Detection Performance
    | Timeline | Threat Actors

Run:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import os
import json
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Telecom Security Analytics",
    page_icon="🛡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Locate the project root robustly (works whether launched from project root or dashboard/)
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent if HERE.name == "dashboard" else HERE
REPORTS = ROOT / "reports"


# ---------------------------------------------------------------------------
# Data loaders (cached)
# ---------------------------------------------------------------------------
@st.cache_data
def load_json(path: Path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_csv(path: Path):
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def load_incidents() -> list:
    """Load the final enriched + ATT&CK-tagged incidents."""
    return load_json(REPORTS / "mitre_tagged_incidents.json") or []


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
def render_sidebar() -> str:
    with st.sidebar:
        st.markdown("## 🛡 Telecom Security Analytics")
        st.markdown("*Multi-Source Telemetry for C2 & Lateral Movement*")
        st.markdown("---")

        page = st.radio(
            "Navigation",
            [
                "Overview",
                "Incident Queue",
                "Incident Detail",
                "Detection Performance",
                "Timeline",
                "Threat Actors",
            ],
            index=0,
        )

        st.markdown("---")
        st.caption(
            "**Data sources**\n\n"
            "- NetFlow\n"
            "- DNS logs\n"
            "- Authentication logs\n"
            "- Threat intelligence"
        )
        st.caption(
            "**Detection models**\n\n"
            "- Isolation Forest (C2)\n"
            "- Random Forest (DNS)\n"
            "- Rule-based (auth)"
        )
        st.markdown("---")
        st.caption("SAS821S Security Analytics Capstone")
    return page


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def tactic_badge(tactic: str) -> str:
    """Return a coloured pill string for a tactic name."""
    colors = {
        "Command and Control":            "#d62728",
        "Exfiltration":                   "#ff7f0e",
        "Lateral Movement":               "#9467bd",
        "Defense Evasion, Persistence":   "#2ca02c",
        "Credential Access":              "#8c564b",
    }
    c = colors.get(tactic, "#7f7f7f")
    return f"<span style='background:{c};color:white;padding:2px 8px;border-radius:10px;font-size:0.8em'>{tactic}</span>"


def detector_badge(det: str) -> str:
    colors = {
        "c2_anomaly":     "#1f77b4",
        "dns_classifier": "#2ca02c",
        "auth_lateral":   "#9467bd",
    }
    c = colors.get(det, "#7f7f7f")
    return f"<span style='background:{c};color:white;padding:2px 8px;border-radius:10px;font-size:0.8em'>{det}</span>"


# ===========================================================================
# PAGE: OVERVIEW
# ===========================================================================
def page_overview(incidents: list):
    st.title("🛡 Security Operations Overview")
    st.markdown(
        "Multi-source telemetry analysis for C2 beaconing and lateral movement "
        "in a telecom network. 7-day synthetic capture, four data sources, three "
        "detection models."
    )
    st.markdown("---")

    if not incidents:
        st.warning("No incidents found. Run the detection pipeline first.")
        return

    # --- Top-line metrics ---
    n_incidents     = len(incidents)
    n_ioc_matched   = sum(1 for i in incidents
                          if i["threat_intel"]["matched_ioc_count"] > 0)
    actors_seen     = sorted({a for i in incidents
                              for a in i["threat_intel"]["threat_actors"]})
    tactics_seen    = sorted({t["tactic"] for i in incidents
                              for t in i["mitre_techniques"]})
    max_final_score = max(i["threat_intel"]["final_score"] for i in incidents)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Incidents", f"{n_incidents}", help="Correlated incidents in queue")
    c2.metric("IOC matched", f"{n_ioc_matched}",
              f"{100*n_ioc_matched/n_incidents:.1f}%")
    c3.metric("Tactics observed", f"{len(tactics_seen)}")
    c4.metric("Threat actors", f"{len(actors_seen)}")
    c5.metric("Max final score", f"{max_final_score:.1f}")

    st.markdown("---")

    # --- Two-column layout: detector distribution + tactic coverage ---
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Detector co-occurrence")
        detector_counts = {}
        for inc in incidents:
            dets = tuple(sorted(inc.get("detectors", [])))
            detector_counts[dets] = detector_counts.get(dets, 0) + 1

        # Top detector combos
        top = sorted(detector_counts.items(), key=lambda x: -x[1])[:8]
        labels = [" + ".join(d) if d else "(none)" for d, _ in top]
        values = [v for _, v in top]

        fig, ax = plt.subplots(figsize=(6, 4))
        y = np.arange(len(labels))
        ax.barh(y, values, color="#1f77b4")
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Incidents")
        ax.set_title("Incidents by detector combination", fontsize=10)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col_right:
        st.subheader("ATT&CK tactic coverage")
        tactic_counts = {}
        for inc in incidents:
            for tactic in set(inc["mitre_tactics"]):
                tactic_counts[tactic] = tactic_counts.get(tactic, 0) + 1

        top_t = sorted(tactic_counts.items(), key=lambda x: -x[1])
        labels = [t for t, _ in top_t]
        values = [v for _, v in top_t]

        fig, ax = plt.subplots(figsize=(6, 4))
        y = np.arange(len(labels))
        ax.barh(y, values, color="#d62728")
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("Incidents")
        ax.set_title("Incidents by ATT&CK tactic", fontsize=10)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.markdown("---")

    # --- Top 15 incidents table ---
    st.subheader("Top 15 incidents by final score")
    rows = []
    for inc in incidents[:15]:
        rows.append({
            "Host":         inc["host"].replace("host:", ""),
            "Window":       inc["window_start"][:16].replace("T", " "),
            "Score":        round(inc["threat_intel"]["final_score"], 1),
            "Detectors":    ", ".join(inc.get("detectors", [])),
            "IOCs":         inc["threat_intel"]["matched_ioc_count"],
            "Actors":       ", ".join(inc["threat_intel"]["threat_actors"]) or "—",
            "Techniques":   len(inc["mitre_techniques"]),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ===========================================================================
# PAGE: DETECTION PERFORMANCE
# ===========================================================================
def page_detection_performance():
    st.title("📊 Detection Performance")
    st.markdown("Evaluation metrics for the three detection models.")
    st.markdown("---")

    # --- DNS classifier ---
    st.subheader("DNS Random Forest classifier")
    dns_metrics = load_json(REPORTS / "dns_metrics.json")
    if dns_metrics:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Precision", f"{dns_metrics['precision']:.3f}")
        c2.metric("Recall",    f"{dns_metrics['recall']:.3f}")
        c3.metric("F1",        f"{dns_metrics['f1']:.3f}")
        c4.metric("ROC-AUC",   f"{dns_metrics['roc_auc']:.3f}")

        st.caption(
            f"Train set: {dns_metrics['n_train']:,} queries • "
            f"Test set: {dns_metrics['n_test']:,} queries • "
            f"Malicious: {dns_metrics['n_positive']:,}"
        )

        st.markdown("**Feature importances**")
        fi = load_csv(REPORTS / "dns_feature_importance.csv")
        if fi is not None:
            fig, ax = plt.subplots(figsize=(8, 3.5))
            ax.barh(fi["feature"][::-1], fi["importance"][::-1], color="#2ca02c")
            ax.set_xlabel("Importance")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
    else:
        st.info("dns_metrics.json not found.")

    st.markdown("---")

    # --- Isolation Forest ---
    st.subheader("Isolation Forest — C2 anomaly detection")
    iso = load_json(REPORTS / "anomaly_metrics.json")
    if iso:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Precision", f"{iso['precision']:.3f}")
        c2.metric("Recall",    f"{iso['recall']:.3f}")
        c3.metric("F1",        f"{iso['f1']:.3f}")
        c4.metric("ROC-AUC",   f"{iso['roc_auc']:.3f}")

        if "precision_at_k" in iso:
            st.markdown("**Precision@K — analyst review depth**")
            pak = iso["precision_at_k"]
            ks = sorted(int(k.replace("top_", "")) for k in pak.keys())
            precs = [pak[f"top_{k}"]["precision"] for k in ks]
            recs  = [pak[f"top_{k}"]["recall"]    for k in ks]

            fig, ax = plt.subplots(figsize=(8, 3.5))
            ax.plot(ks, precs, "o-", label="Precision@K", color="#d62728")
            ax.plot(ks, recs,  "s-", label="Recall@K",    color="#1f77b4")
            if "random_baseline_precision" in iso:
                ax.axhline(iso["random_baseline_precision"],
                           color="grey", linestyle="--", label="Random baseline")
            ax.set_xscale("log")
            ax.set_xlabel("K (top-ranked alerts reviewed)")
            ax.set_ylabel("Score")
            ax.legend()
            ax.grid(alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
    else:
        st.info("anomaly_metrics.json not found.")

    st.markdown("---")

    # --- Auth analytics ---
    st.subheader("Authentication analytics")
    auth_s = load_json(REPORTS / "auth_summary.json")
    if auth_s:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Users",            f"{auth_s['n_users']}")
        c2.metric("Windows scored",   f"{auth_s['n_windows']:,}")
        c3.metric("Incidents",        f"{auth_s['n_incidents']}")
        c4.metric("High-risk (≥60)",  f"{auth_s['n_high_risk']}")
        st.caption(f"Top incident users: {', '.join(auth_s['top_incident_users'])}")
    else:
        st.info("auth_summary.json not found.")


# ===========================================================================
# PAGE: INCIDENT QUEUE
# ===========================================================================
def page_incident_queue(incidents: list):
    st.title("📋 Incident Queue")
    st.markdown(
        "Sortable, filterable list of all correlated incidents — the primary "
        "triage view for the SOC analyst."
    )
    st.markdown("---")

    if not incidents:
        st.warning("No incidents found.")
        return

    # --- Build a flat DataFrame for filtering/sorting ---
    df = pd.DataFrame([{
        "host":           inc["host"].replace("host:", ""),
        "window_start":   inc["window_start"],
        "window_end":     inc["window_end"],
        "final_score":    inc["threat_intel"]["final_score"],
        "composite":      inc["composite_score"],
        "ti_score":       inc["threat_intel"]["ti_score"],
        "detector_count": inc["detector_count"],
        "detectors":      ", ".join(inc.get("detectors", [])),
        "ioc_count":      inc["threat_intel"]["matched_ioc_count"],
        "actors":         ", ".join(inc["threat_intel"]["threat_actors"]) or "—",
        "tactic_count":   len(inc["mitre_tactics"]),
        "technique_count": len(inc["mitre_techniques"]),
        "escalation":     inc["components"].get("escalation", 0),
    } for inc in incidents])

    # --- Filter controls ---
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        min_score = st.slider("Min final score", 0, 150, 50, step=5)
    with col2:
        detector_filter = st.selectbox(
            "Detector",
            ["(any)", "c2_anomaly", "dns_classifier", "auth_lateral"],
        )
    with col3:
        actor_filter = st.selectbox(
            "Threat actor",
            ["(any)"] + sorted({a for i in incidents
                                for a in i["threat_intel"]["threat_actors"]}),
        )
    with col4:
        min_iocs = st.slider("Min IOC matches", 0, 5, 0, step=1)

    # --- Apply filters ---
    filtered = df[df["final_score"] >= min_score].copy()
    if detector_filter != "(any)":
        filtered = filtered[filtered["detectors"].str.contains(detector_filter)]
    if actor_filter != "(any)":
        filtered = filtered[filtered["actors"].str.contains(actor_filter)]
    if min_iocs > 0:
        filtered = filtered[filtered["ioc_count"] >= min_iocs]

    # --- Sort control ---
    sort_col = st.selectbox(
        "Sort by",
        ["final_score", "ioc_count", "detector_count",
         "tactic_count", "technique_count"],
        index=0,
    )
    filtered = filtered.sort_values(sort_col, ascending=False).reset_index(drop=True)

    st.markdown(f"**{len(filtered)} incidents match the current filters**")

    # --- Display table ---
    st.dataframe(
        filtered[[
            "host", "window_start", "final_score", "detector_count",
            "detectors", "ioc_count", "actors",
            "tactic_count", "technique_count",
        ]],
        use_container_width=True,
        hide_index=True,
        column_config={
            "final_score":    st.column_config.ProgressColumn(
                "Final score", min_value=0, max_value=150, format="%.1f"
            ),
            "ioc_count":      st.column_config.NumberColumn("IOCs", format="%d"),
            "detector_count": st.column_config.NumberColumn("Detectors", format="%d"),
            "tactic_count":   st.column_config.NumberColumn("Tactics", format="%d"),
            "technique_count": st.column_config.NumberColumn("Techniques", format="%d"),
        },
    )


# ===========================================================================
# PAGE: INCIDENT DETAIL
# ===========================================================================
def page_incident_detail(incidents: list):
    st.title("🔍 Incident Detail")
    st.markdown(
        "Deep-dive view: composite scoring, evidence from each detector, "
        "threat-intelligence context, and ATT&CK technique attribution."
    )
    st.markdown("---")

    if not incidents:
        st.warning("No incidents found.")
        return

    # --- Incident selector ---
    options = [
        f"{inc['host']}  |  {inc['window_start'][:16].replace('T',' ')}  "
        f"|  score={inc['threat_intel']['final_score']:.1f}"
        for inc in incidents[:100]
    ]
    selected_idx = st.selectbox(
        "Select incident (top 100 shown, sorted by final score)",
        range(len(options)),
        format_func=lambda i: options[i],
    )
    inc = incidents[selected_idx]

    st.markdown("---")

    # --- Header metrics ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Final score", f"{inc['threat_intel']['final_score']:.1f}")
    c2.metric("Composite",   f"{inc['composite_score']:.1f}")
    c3.metric("TI bonus",    f"+{inc['threat_intel']['ti_score']:.1f}")
    c4.metric("Detectors",   f"{inc['detector_count']}")

    # --- Host / window info ---
    st.markdown(
        f"**Host:** `{inc['host'].replace('host:', '')}`  &nbsp;|&nbsp;  "
        f"**Window:** {inc['window_start'][:19].replace('T', ' ')} → "
        f"{inc['window_end'][11:19]}"
    )

    # --- Detectors ---
    st.markdown("**Detectors fired**")
    badges = " ".join(detector_badge(d) for d in inc.get("detectors", []))
    st.markdown(badges, unsafe_allow_html=True)

    # --- Composite score breakdown ---
    st.markdown("---")
    st.subheader("Score breakdown")
    comp = inc["components"]
    comp_df = pd.DataFrame({
        "Component": [
            "C2 anomaly (Isolation Forest)",
            "DNS classifier (Random Forest)",
            "Auth lateral movement",
            "Multi-detector escalation",
            "Threat-intel enrichment",
        ],
        "Contribution": [
            comp.get("c2", 0),
            comp.get("dns", 0),
            comp.get("auth", 0),
            comp.get("escalation", 0),
            inc["threat_intel"]["ti_score"],
        ],
        "Max": [40, 35, 25, 15, 15],
    })
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    # --- ATT&CK techniques ---
    st.markdown("---")
    st.subheader("MITRE ATT&CK techniques")
    tech_df = pd.DataFrame([{
        "ID":         t["id"],
        "Name":       t["name"],
        "Tactic":     t["tactic"],
        "Confidence": t["confidence"],
        "Source":     t["source"],
    } for t in inc["mitre_techniques"]])
    st.dataframe(tech_df, use_container_width=True, hide_index=True)

    # --- Threat intel ---
    st.markdown("---")
    st.subheader("Threat intelligence enrichment")
    ti = inc["threat_intel"]

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown(f"**Matched IOCs** ({ti['matched_ioc_count']})")
        if ti["matched_iocs"]:
            ioc_df = pd.DataFrame(ti["matched_iocs"])
            st.dataframe(
                ioc_df[["ioc_type", "value", "threat_actor", "confidence"]],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No IOC matches for this incident.")

    with col_right:
        st.markdown("**Similar known attack patterns**")
        if ti["similar_attacks"]:
            sim_df = pd.DataFrame(ti["similar_attacks"])[
                ["name", "similarity"]
            ]
            sim_df["similarity"] = sim_df["similarity"].round(3)
            st.dataframe(sim_df, use_container_width=True, hide_index=True)
        else:
            st.info("No similar patterns above similarity threshold.")

    # --- Evidence from each detector ---
    st.markdown("---")
    st.subheader("Detector evidence")
    for det_name, ev in inc.get("evidence", {}).items():
        with st.expander(f"📌 {det_name}", expanded=True):
            st.json(ev)


# ===========================================================================
# PAGE: TIMELINE
# ===========================================================================
def page_timeline(incidents: list):
    st.title("🕒 Incident Timeline")
    st.markdown(
        "Chronological view of incidents across the 7-day capture period. "
        "The injected beacon + lateral-movement chain appears daily at 14:00."
    )
    st.markdown("---")

    if not incidents:
        st.warning("No incidents found.")
        return

    # --- Build timeline DataFrame ---
    df = pd.DataFrame([{
        "window_start": pd.to_datetime(inc["window_start"]),
        "host":         inc["host"].replace("host:", ""),
        "final_score":  inc["threat_intel"]["final_score"],
        "n_detectors":  inc["detector_count"],
        "n_iocs":       inc["threat_intel"]["matched_ioc_count"],
        "n_tactics":    len(inc["mitre_tactics"]),
    } for inc in incidents])

    # --- Filter by score threshold ---
    min_score = st.slider("Minimum final score", 0, 150, 80, step=5)
    df = df[df["final_score"] >= min_score].copy()

    if df.empty:
        st.info("No incidents above the selected threshold.")
        return

    # --- Scatter: score over time, size = detector count, colour = IOC count ---
    fig, ax = plt.subplots(figsize=(14, 5))

    # Group by IOC count for colour legend
    for ioc_count in sorted(df["n_iocs"].unique()):
        sub = df[df["n_iocs"] == ioc_count]
        ax.scatter(
            sub["window_start"], sub["final_score"],
            s=30 + sub["n_detectors"] * 30,
            alpha=0.65,
            label=f"IOCs matched: {ioc_count}",
        )

    ax.set_xlabel("Time")
    ax.set_ylabel("Final score")
    ax.set_title(f"Incident timeline (threshold ≥ {min_score})", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown("---")

    # --- Daily incident count ---
    st.subheader("Incidents per day")
    df["day"] = df["window_start"].dt.date
    daily = df.groupby("day").size().reset_index(name="incidents")

    fig, ax = plt.subplots(figsize=(14, 3))
    ax.bar(daily["day"].astype(str), daily["incidents"], color="#2ca02c")
    ax.set_ylabel("Incidents")
    ax.set_xlabel("Day")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown("---")

    # --- Host frequency table ---
    st.subheader("Most frequent hosts in incident queue")
    host_counts = (
        df.groupby("host")
          .agg(
              incidents=("final_score", "count"),
              mean_score=("final_score", "mean"),
              max_score=("final_score", "max"),
              total_iocs=("n_iocs", "sum"),
          )
          .sort_values("incidents", ascending=False)
          .head(15)
          .reset_index()
    )
    host_counts["mean_score"] = host_counts["mean_score"].round(1)
    host_counts["max_score"]  = host_counts["max_score"].round(1)
    st.dataframe(host_counts, use_container_width=True, hide_index=True)


# ===========================================================================
# PAGE: THREAT ACTORS
# ===========================================================================
def page_threat_actors(incidents: list):
    st.title("🎯 Threat Actor Attribution")
    st.markdown(
        "Attribution analysis based on IOC matches against the threat "
        "intelligence feed."
    )
    st.markdown("---")

    if not incidents:
        st.warning("No incidents found.")
        return

    # --- Aggregate by actor ---
    actor_stats = {}
    for inc in incidents:
        for actor in inc["threat_intel"]["threat_actors"]:
            if actor not in actor_stats:
                actor_stats[actor] = {
                    "incidents":  0,
                    "max_score":  0,
                    "hosts":      set(),
                    "tactics":    set(),
                    "techniques": set(),
                }
            s = actor_stats[actor]
            s["incidents"] += 1
            s["max_score"] = max(s["max_score"], inc["threat_intel"]["final_score"])
            s["hosts"].add(inc["host"].replace("host:", ""))
            s["tactics"].update(inc["mitre_tactics"])
            s["techniques"].update(t["id"] for t in inc["mitre_techniques"])

    if not actor_stats:
        st.info("No actor attribution found in the incident queue.")
        return

    # --- Metric tiles per actor ---
    actors_sorted = sorted(actor_stats.items(), key=lambda x: -x[1]["incidents"])
    cols = st.columns(len(actors_sorted))
    for col, (actor, s) in zip(cols, actors_sorted):
        col.metric(
            f"{actor}",
            f"{s['incidents']} incidents",
            f"max score {s['max_score']:.0f}",
        )

    st.markdown("---")

    # --- Summary table ---
    st.subheader("Attribution summary")
    summary_rows = [{
        "Actor":           actor,
        "Incidents":       s["incidents"],
        "Unique hosts":    len(s["hosts"]),
        "Distinct tactics": len(s["tactics"]),
        "Distinct techniques": len(s["techniques"]),
        "Max final score": round(s["max_score"], 1),
    } for actor, s in actors_sorted]
    st.dataframe(pd.DataFrame(summary_rows),
                 use_container_width=True, hide_index=True)

    st.markdown("---")

    # --- Two-column: hosts + techniques per actor ---
    for actor, s in actors_sorted:
        with st.expander(f"🔎 {actor} — detail", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Hosts observed**")
                st.write(sorted(s["hosts"])[:20])
            with c2:
                st.markdown("**Tactics & techniques**")
                st.write(f"Tactics: {sorted(s['tactics'])}")
                st.write(f"Techniques: {sorted(s['techniques'])}")

    st.markdown("---")

    # --- Actor comparison bar chart ---
    st.subheader("Actor incident volume")
    labels = [a for a, _ in actors_sorted]
    values = [s["incidents"] for _, s in actors_sorted]

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(labels, values, color=["#1f77b4", "#d62728", "#2ca02c", "#9467bd"][:len(labels)])
    ax.set_ylabel("Incidents")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ===========================================================================
# MAIN DISPATCHER
# ===========================================================================
def main():
    # Sidebar navigation returns the current page name
    page = render_sidebar()

    # Load data once; cached across page switches
    incidents = load_incidents()

    # Dispatch
    if page == "Overview":
        page_overview(incidents)
    elif page == "Incident Queue":
        page_incident_queue(incidents)
    elif page == "Incident Detail":
        page_incident_detail(incidents)
    elif page == "Detection Performance":
        page_detection_performance()
    elif page == "Timeline":
        page_timeline(incidents)
    elif page == "Threat Actors":
        page_threat_actors(incidents)
    else:
        st.error(f"Unknown page: {page}")


if __name__ == "__main__":
    main()
