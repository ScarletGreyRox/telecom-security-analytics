"""
run_pipeline.py
---------------
End-to-end pipeline driver for the SAS821S Capstone Project.

Runs every stage in order:
    1. Data generation    (synthetic multi-source telemetry)
    2. Feature engineering
    3. DNS classifier
    4. C2 anomaly detection (Isolation Forest)
    5. Auth analytics (lateral movement)
    6. Event correlation
    7. Threat intelligence enrichment
    8. MITRE ATT&CK mapping

Usage:
    python run_pipeline.py
    python run_pipeline.py --skip-data       (reuse existing raw data)
    python run_pipeline.py --start-from features
"""

from __future__ import annotations

import os
import sys
import time
import argparse
from pathlib import Path

# Make src/ importable regardless of invocation directory
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))


# ---------------------------------------------------------------------------
# Stage definitions
# ---------------------------------------------------------------------------
STAGES = ["data", "features", "dns", "anomaly", "auth",
          "correlation", "threat_intel", "mitre"]


def _banner(msg: str) -> None:
    line = "=" * 70
    print()
    print(line)
    print(f"  {msg}")
    print(line)


# ---------------------------------------------------------------------------
# Individual stages
# ---------------------------------------------------------------------------
def stage_data():
    _banner("STAGE 1/8  —  Data generation")
    import data_generator as dg
    netflow, dns, auth = dg.build_dataset(inject_attacks=True)
    dg.save_all(netflow, dns, auth, out_dir=str(ROOT / "data" / "raw"))


def stage_features():
    _banner("STAGE 2/8  —  Feature engineering")
    import pandas as pd
    import features as ft

    raw = ROOT / "data" / "raw"
    netflow = pd.read_csv(raw / "netflow.csv", parse_dates=["timestamp"])
    dns     = pd.read_csv(raw / "dns.csv",     parse_dates=["timestamp"])
    auth    = pd.read_csv(raw / "auth.csv",    parse_dates=["timestamp"])

    feats = ft.build_features(netflow, dns, auth, window="1h")

    out = ROOT / "data" / "processed" / "features.pkl"
    out.parent.mkdir(parents=True, exist_ok=True)
    feats.to_pickle(out)
    print(f"✔ Saved: {out}")


def stage_dns():
    _banner("STAGE 3/8  —  DNS classifier")
    import pandas as pd
    import dns_classifier as dc

    dns = pd.read_csv(ROOT / "data" / "raw" / "dns.csv",
                      parse_dates=["timestamp"])
    result = dc.train_dns_classifier(dns)
    dc.save_artifacts(result,
                      model_dir=str(ROOT / "models"),
                      report_dir=str(ROOT / "reports"))


def stage_anomaly():
    _banner("STAGE 4/8  —  C2 anomaly detection (Isolation Forest)")
    import pandas as pd
    import anomaly_detector as ad

    feats = pd.read_pickle(ROOT / "data" / "processed" / "features.pkl")
    result = ad.train_isolation_forest(feats, contamination=0.05)
    ad.save_artifacts(result,
                      model_dir=str(ROOT / "models"),
                      report_dir=str(ROOT / "reports"))


def stage_auth():
    _banner("STAGE 5/8  —  Auth analytics (lateral movement)")
    import pandas as pd
    import auth_analytics as aa

    auth    = pd.read_csv(ROOT / "data" / "raw" / "auth.csv",
                          parse_dates=["timestamp"])
    netflow = pd.read_csv(ROOT / "data" / "raw" / "netflow.csv",
                          parse_dates=["timestamp"])
    aa.run_auth_analytics(auth, netflow)


def stage_correlation():
    _banner("STAGE 6/8  —  Event correlation")
    import correlation as co
    co.run_correlation()


def stage_threat_intel():
    _banner("STAGE 7/8  —  Threat intelligence enrichment")
    import threat_intelligence as ti
    ti.run_threat_intelligence()


def stage_mitre():
    _banner("STAGE 8/8  —  MITRE ATT&CK mapping")
    import mitre_mapping as mm
    mm.run_mitre_mapping()


STAGE_FUNCS = {
    "data":        stage_data,
    "features":    stage_features,
    "dns":         stage_dns,
    "anomaly":     stage_anomaly,
    "auth":        stage_auth,
    "correlation": stage_correlation,
    "threat_intel": stage_threat_intel,
    "mitre":       stage_mitre,
}


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="SAS821S Capstone pipeline driver")
    parser.add_argument("--skip-data", action="store_true",
                        help="Skip the data generation stage (reuse existing CSVs)")
    parser.add_argument("--start-from", choices=STAGES, default="data",
                        help="Start from a specific stage")
    args = parser.parse_args()

    stages_to_run = list(STAGES)
    if args.skip_data and "data" in stages_to_run:
        stages_to_run.remove("data")
    if args.start_from in stages_to_run:
        idx = stages_to_run.index(args.start_from)
        stages_to_run = stages_to_run[idx:]

    print(f"Pipeline will run: {' -> '.join(stages_to_run)}")
    t0 = time.time()

    for stage in stages_to_run:
        ts = time.time()
        try:
            STAGE_FUNCS[stage]()
        except Exception as e:
            print(f"\n✘ Stage '{stage}' FAILED: {e}")
            raise
        print(f"  → Stage '{stage}' completed in {time.time() - ts:.1f}s")

    total = time.time() - t0
    _banner(f"Pipeline complete in {total:.1f}s")
    print("Outputs:")
    print(f"  data/raw/         : CSV telemetry")
    print(f"  data/processed/   : features.pkl")
    print(f"  models/           : dns_rf.pkl, iso_forest.pkl")
    print(f"  reports/          : JSON metrics + incident lists")
    print()


if __name__ == "__main__":
    main()
