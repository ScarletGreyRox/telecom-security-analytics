# Multi-Source Telemetry Analytics for Detecting C2 Beacons and Lateral Movement in Telecom Networks

**SAS821S — Security Analytics Capstone Project**

---

## Overview

A Python-based security analytics pipeline that ingests network, DNS, authentication, and threat-intelligence telemetry, engineers behavioural features, and applies machine learning and rule-based detection to identify three classes of attacker behaviour:

1. **C2 beaconing** — periodic low-volume callbacks (Isolation Forest)
2. **DNS tunnelling** — high-entropy subdomain queries (Random Forest)
3. **Lateral movement** — authentication fan-out with SMB pivoting (rule-based)

Detections are correlated across telemetry sources, enriched with threat intelligence, mapped to MITRE ATT&CK techniques, and presented through an interactive Streamlit SOC dashboard.

---

## Architecture

```
NetFlow ─┐
DNS     ─┼─► Feature Engineering ─► Detectors ─► Correlation ─► TI Enrichment ─► ATT&CK ─► Dashboard
Auth    ─┘                                                                        │
Threat Intel ─────────────────────────────────────────────────────────────────────┘
```

| # | Stage | Module | Output |
|---|---|---|---|
| 1 | Data generation | `src/data_generator.py` | `data/raw/*.csv` |
| 2 | Feature engineering | `src/features.py` | `data/processed/features.pkl` |
| 3 | DNS classifier | `src/dns_classifier.py` | `models/dns_rf.pkl`, `reports/dns_*.json` |
| 4 | C2 anomaly detection | `src/anomaly_detector.py` | `models/iso_forest.pkl`, `reports/anomaly_*.json` |
| 5 | Auth analytics | `src/auth_analytics.py` | `reports/lateral_incidents.json` |
| 6 | Event correlation | `src/correlation.py` | `reports/correlated_incidents.json` |
| 7 | Threat intelligence | `src/threat_intelligence.py` | `reports/enriched_incidents.json` |
| 8 | MITRE mapping | `src/mitre_mapping.py` | `reports/mitre_tagged_incidents.json` |

---

## Project structure

```
Telecom-Security-Analytics/
├── dashboard/app.py            Streamlit SOC dashboard
├── data/
│   ├── raw/                    Generated telemetry
│   ├── processed/              features.pkl
│   └── threat_intelligence/    IOC feed + patterns
├── models/                     Trained models
├── notebooks/                  Exploration notebooks
├── reports/                    Metrics, incidents, screenshots
├── src/                        Pipeline modules (8 stages)
├── tests/                      pytest suite (100 fast + 4 slow)
├── run_pipeline.py             End-to-end driver
├── pytest.ini
├── requirements.txt
└── README.md
```

---

## Installation

```bash
pip install -r requirements.txt
```

**Python:** 3.10+ (developed on 3.14)

---

## Running

**Full pipeline:**
```bash
python run_pipeline.py
```

**Skip data generation:**
```bash
python run_pipeline.py --skip-data
```

**Resume from a specific stage:**
```bash
python run_pipeline.py --start-from correlation
```

**Dashboard** (after the pipeline has produced its reports):
```bash
streamlit run dashboard/app.py
```
Then open http://localhost:8501

---

## Running the tests

```bash
python -m pytest -v -m "not slow"     # fast suite (~80s, 100 tests)
python -m pytest -v -m slow           # integration tests (~90s, 4 tests)
```

---

## Detection models

### DNS Random Forest classifier
- 8 subdomain-behavioural features (entropy, length, digit ratio, query type mix)
- Base-domain identity features deliberately excluded to prevent shortcut learning
- **Result:** F1 = 0.75, Precision = 0.60, Recall = 1.00, ROC-AUC = 0.98

### C2 Isolation Forest
- 6 behavioural features (regularity, delta_cv, top_dst_share, top_port_share, event_count, avg_bytes_sent)
- **Result:** Precision@10 = 1.00, Precision@25 = 1.00, Precision@50 = 0.84

### Auth lateral movement detector
- Rule: `unique_hosts >= 3` AND `target_rate >= 0.5` in 15 min, corroborated by SMB/RDP from the same pivot host
- **Result:** Injected user007 chains dominate 6 of top-7 incidents

---

## Dataset

Synthetic 7-day capture starting **2026-08-10** (Monday):

- **NetFlow:** ~67,000 flows with diurnal + weekend patterns
- **DNS:** ~38,000 queries, ~10% malicious
- **Auth:** ~10,600 events across 60 users

**Injected attack scenarios:**

| Scenario | Host | Signature |
|---|---|---|
| C2 beacon | `10.0.1.50` | HTTPS every 60s, 6h/day, to `172.16.99.10` |
| Lateral movement | `10.0.1.50` (as `user007`) | 4 hosts in 15 min, SMB pivots |
| DNS tunnel | `10.0.2.77` | 40-char random subdomains to `dns-tunnel.xyz` |
| Exfiltration | `10.0.3.111` | 5-50 MB transfers at 02:00 to `172.16.99.20` |

---

## Dashboard pages

| Page | Purpose |
|---|---|
| Overview | Metrics, detector co-occurrence, ATT&CK tactic coverage |
| Incident Queue | Filterable, sortable incident list |
| Incident Detail | Score breakdown, IOCs, ATT&CK, evidence |
| Detection Performance | Model metrics for all three detectors |
| Timeline | Temporal distribution of incidents |
| Threat Actors | Attribution breakdown (APT-SIM-01, APT-SIM-02) |

---

## Test coverage

**100 fast tests across 8 files + 4 slow integration tests:**

Six regression guards lock in the design iterations:
1. Base-domain feature leakage in DNS classifier
2. Time-of-day pollution in anomaly detector
3. Host-scoped auth correlation (`pivot_hosts`)
4. Lateral-movement detection (`user007` in top-10)
5. Correlation dominance (`10.0.1.50` in top-5)
6. Population multi-detector rate (< 30%)

---

## References

- MITRE ATT&CK Framework — https://attack.mitre.org/
- Isolation Forest — Liu, Ting, Zhou (2008)
- Random Forest — Breiman (2001)
- Shannon entropy for DNS tunnelling detection
