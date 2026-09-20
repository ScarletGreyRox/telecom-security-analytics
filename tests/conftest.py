"""
Shared pytest fixtures for the SAS821S Capstone test suite.

Fixtures load pre-built artifacts (fast) so the suite runs in seconds.
The integration test (marked `slow`) regenerates everything.
"""

import os
import json
import sys
import pytest
import pandas as pd

# Make src/ importable from tests/
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))


@pytest.fixture(scope="session")
def project_root():
    return ROOT


@pytest.fixture(scope="session")
def data_dir(project_root):
    return os.path.join(project_root, "data")


@pytest.fixture(scope="session")
def reports_dir(project_root):
    return os.path.join(project_root, "reports")


@pytest.fixture(scope="session")
def raw_netflow(data_dir):
    p = os.path.join(data_dir, "raw", "netflow.csv")
    if not os.path.exists(p):
        pytest.skip("netflow.csv not found — run data_generator first")
    return pd.read_csv(p, parse_dates=["timestamp"])


@pytest.fixture(scope="session")
def raw_dns(data_dir):
    p = os.path.join(data_dir, "raw", "dns.csv")
    if not os.path.exists(p):
        pytest.skip("dns.csv not found")
    return pd.read_csv(p, parse_dates=["timestamp"])


@pytest.fixture(scope="session")
def raw_auth(data_dir):
    p = os.path.join(data_dir, "raw", "auth.csv")
    if not os.path.exists(p):
        pytest.skip("auth.csv not found")
    return pd.read_csv(p, parse_dates=["timestamp"])


@pytest.fixture(scope="session")
def features_table(data_dir):
    p = os.path.join(data_dir, "processed", "features.pkl")
    if not os.path.exists(p):
        pytest.skip("features.pkl not found — run features.py first")
    return pd.read_pickle(p)


@pytest.fixture(scope="session")
def correlated_incidents(reports_dir):
    p = os.path.join(reports_dir, "correlated_incidents.json")
    if not os.path.exists(p):
        pytest.skip("correlated_incidents.json not found")
    with open(p) as f:
        return json.load(f)


@pytest.fixture(scope="session")
def enriched_incidents(reports_dir):
    p = os.path.join(reports_dir, "enriched_incidents.json")
    if not os.path.exists(p):
        pytest.skip("enriched_incidents.json not found")
    with open(p) as f:
        return json.load(f)


@pytest.fixture(scope="session")
def tagged_incidents(reports_dir):
    p = os.path.join(reports_dir, "mitre_tagged_incidents.json")
    if not os.path.exists(p):
        pytest.skip("mitre_tagged_incidents.json not found")
    with open(p) as f:
        return json.load(f)


@pytest.fixture(scope="session")
def lateral_incidents(reports_dir):
    p = os.path.join(reports_dir, "lateral_incidents.json")
    if not os.path.exists(p):
        pytest.skip("lateral_incidents.json not found")
    with open(p) as f:
        return json.load(f)
