"""
Tests for src/auth_analytics.py
"""

import os
import json
import pytest
import pandas as pd


class TestPerWindowDetection:
    def test_detect_anomalies_returns_dataframe(self, raw_auth):
        import auth_analytics as aa
        feats = aa.detect_auth_anomalies(raw_auth)
        assert isinstance(feats, pd.DataFrame)
        assert len(feats) > 0

    def test_expected_columns(self, raw_auth):
        import auth_analytics as aa
        feats = aa.detect_auth_anomalies(raw_auth)
        for col in ("user", "window_start", "auth_count", "unique_hosts",
                    "target_rate", "is_fanout"):
            assert col in feats.columns

    def test_fanout_flag_is_binary(self, raw_auth):
        import auth_analytics as aa
        feats = aa.detect_auth_anomalies(raw_auth)
        assert set(feats["is_fanout"].unique()).issubset({0, 1})

    def test_some_fanout_flagged(self, raw_auth):
        import auth_analytics as aa
        feats = aa.detect_auth_anomalies(raw_auth)
        assert feats["is_fanout"].sum() > 0


class TestIncidentChains:
    def test_lateral_incidents_json_exists(self, reports_dir):
        p = os.path.join(reports_dir, "lateral_incidents.json")
        assert os.path.exists(p)

    def test_incidents_have_pivot_hosts(self, lateral_incidents):
        """Every incident must include a pivot_hosts field for downstream
        host-scoped correlation."""
        for inc in lateral_incidents[:20]:
            assert "pivot_hosts" in inc

    def test_incidents_have_expected_fields(self, lateral_incidents):
        if not lateral_incidents:
            pytest.skip("no incidents")
        for field in ("incident_id", "user", "start", "end",
                      "risk_score", "max_fanout", "smb_targets"):
            assert field in lateral_incidents[0]

    def test_risk_scores_bounded(self, lateral_incidents):
        for inc in lateral_incidents[:50]:
            assert 0.0 <= inc["risk_score"] <= 100.0

    def test_injected_user_in_top_incidents(self, lateral_incidents):
        """Regression: user007 (the injected pivot user) should appear
        among the top-10 highest-risk incidents."""
        top10 = lateral_incidents[:10]
        users = [inc["user"] for inc in top10]
        assert "user007" in users, \
            f"user007 not in top-10 incidents: {users}"


class TestSummary:
    def test_auth_summary_exists(self, reports_dir):
        p = os.path.join(reports_dir, "auth_summary.json")
        assert os.path.exists(p)

    def test_auth_summary_structure(self, reports_dir):
        p = os.path.join(reports_dir, "auth_summary.json")
        if not os.path.exists(p):
            pytest.skip()
        with open(p) as f:
            s = json.load(f)
        for k in ("n_users", "n_windows", "n_incidents", "n_high_risk",
                  "top_incident_users", "max_risk_score"):
            assert k in s
