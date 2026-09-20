"""
Tests for src/dns_classifier.py
"""

import os
import pytest
import numpy as np
import pandas as pd


class TestFeatureExtraction:
    def test_extract_query_features_returns_dataframe(self, raw_dns):
        import dns_classifier as dc
        # Use a small sample for speed
        sample = raw_dns.head(500)
        X = dc.extract_query_features(sample)
        assert isinstance(X, pd.DataFrame)
        assert len(X) == len(sample)

    def test_query_features_no_base_domain_leakage(self, raw_dns):
        """Regression test: base_len / base_entropy / num_labels must NOT
        appear, because they leak the malicious domain identity."""
        import dns_classifier as dc
        X = dc.extract_query_features(raw_dns.head(100))
        forbidden = {"base_len", "base_entropy", "num_labels"}
        leaked = forbidden & set(X.columns)
        assert not leaked, f"Base-domain features leaked into X: {leaked}"

    def test_expected_features_present(self, raw_dns):
        import dns_classifier as dc
        X = dc.extract_query_features(raw_dns.head(100))
        for f in ["sub_len", "sub_entropy", "sub_digit_ratio",
                  "sub_hyphen_count", "is_long_query",
                  "query_type_txt", "query_type_aaaa", "is_nxdomain"]:
            assert f in X.columns, f"Missing feature: {f}"


class TestLabeling:
    def test_labels_are_binary(self, raw_dns):
        import dns_classifier as dc
        X, y = dc.build_labelled_dns(raw_dns.head(500))
        assert set(y.unique()).issubset({0, 1})

    def test_malicious_minority(self, raw_dns):
        import dns_classifier as dc
        X, y = dc.build_labelled_dns(raw_dns)
        frac = y.mean()
        assert 0.01 < frac < 0.30, f"Malicious ratio {frac} unexpected"

    def test_tunnel_domain_is_labelled_malicious(self, raw_dns):
        import dns_classifier as dc
        X, y = dc.build_labelled_dns(raw_dns)
        tunnel_mask = raw_dns["domain"] == "dns-tunnel.xyz"
        assert y[tunnel_mask.values].all(), \
            "dns-tunnel.xyz queries should all be labelled malicious"


class TestModel:
    def test_model_file_exists(self, project_root):
        p = os.path.join(project_root, "models", "dns_rf.pkl")
        assert os.path.exists(p), "dns_rf.pkl not found — run dns_classifier first"

    def test_trained_model_has_expected_feature_count(self, project_root):
        import joblib
        p = os.path.join(project_root, "models", "dns_rf.pkl")
        if not os.path.exists(p):
            pytest.skip("model not trained yet")
        clf = joblib.load(p)
        # We expect 8 features
        assert hasattr(clf, "n_features_in_")
        assert clf.n_features_in_ == 8


class TestMetrics:
    def test_dns_metrics_file_exists(self, reports_dir):
        p = os.path.join(reports_dir, "dns_metrics.json")
        assert os.path.exists(p)

    def test_dns_metrics_in_valid_range(self, reports_dir):
        import json
        p = os.path.join(reports_dir, "dns_metrics.json")
        if not os.path.exists(p):
            pytest.skip("metrics not generated")
        with open(p) as f:
            m = json.load(f)
        for k in ("precision", "recall", "f1", "roc_auc"):
            assert 0.0 <= m[k] <= 1.0
        # Recall was 1.0 in our run — should stay high
        assert m["recall"] > 0.5
        assert m["roc_auc"] > 0.7
