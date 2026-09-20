"""
Tests for src/correlation.py
"""

import os
import json
import pytest
import pandas as pd


class TestLoaders:
    def test_load_anomaly_scores(self, reports_dir):
        import correlation as co
        df = co.load_anomaly_scores(os.path.join(reports_dir, "anomaly_scores.csv"))
        assert "entity" in df.columns
        assert "c2_score_norm" in df.columns

    def test_load_lateral_incidents(self, reports_dir):
        import correlation as co
        inc = co.load_lateral_incidents(
            os.path.join(reports_dir, "lateral_incidents.json"))
        assert isinstance(inc, list)
        assert len(inc) > 0


class TestCorrelatedOutput:
    def test_correlated_incidents_file_exists(self, reports_dir):
        p = os.path.join(reports_dir, "correlated_incidents.json")
        assert os.path.exists(p)

    def test_correlated_incidents_structure(self, correlated_incidents):
        if not correlated_incidents:
            pytest.skip()
        inc = correlated_incidents[0]
        for field in ("host", "window_start", "window_end",
                      "composite_score", "detector_count",
                      "detectors", "components", "evidence"):
            assert field in inc

    def test_composite_score_bounded(self, correlated_incidents):
        """Composite = c2(40) + dns(35) + auth(25) + escalation(15) <= 115"""
        for inc in correlated_incidents:
            assert 0 <= inc["composite_score"] <= 115.0 + 1e-6

    def test_detector_count_bounded(self, correlated_incidents):
        for inc in correlated_incidents:
            assert 0 <= inc["detector_count"] <= 3

    def test_top_incidents_dominated_by_compromised_host(self, correlated_incidents):
        """Regression: our beacon/pivot host 10.0.1.50 should dominate
        the top of the correlated queue."""
        top5_hosts = [inc["host"] for inc in correlated_incidents[:5]]
        assert sum(1 for h in top5_hosts if "10.0.1.50" in h) >= 3, \
            f"Top 5 not dominated by 10.0.1.50: {top5_hosts}"

    def test_multi_detector_rate_is_reasonable(self, reports_dir):
        """Regression: after host-scoped auth fix, multi-detector windows
        should be a small minority *of all scored windows*.

        NOTE: we check correlation_summary.json, not the filtered
        correlated_incidents.json. The incident list is by definition the
        interesting subset (composite score >= 10), so a high multi-detector
        rate there is expected. The population-level rate is what matters.
        """
        p = os.path.join(reports_dir, "correlation_summary.json")
        if not os.path.exists(p):
            pytest.skip()
        with open(p) as f:
            s = json.load(f)
        total = s["n_windows_scored"]
        multi = s["n_multi_detector"]
        frac = multi / total if total > 0 else 0
        assert frac < 0.30, \
            f"Multi-detector fraction {frac:.2%} of all windows is too high - auth spillover?"


class TestSummary:
    def test_correlation_summary_exists(self, reports_dir):
        p = os.path.join(reports_dir, "correlation_summary.json")
        assert os.path.exists(p)

    def test_summary_structure(self, reports_dir):
        p = os.path.join(reports_dir, "correlation_summary.json")
        if not os.path.exists(p):
            pytest.skip()
        with open(p) as f:
            s = json.load(f)
        for k in ("n_windows_scored", "n_high_confidence", "n_multi_detector",
                  "max_composite_score", "detector_count_distribution"):
            assert k in s
