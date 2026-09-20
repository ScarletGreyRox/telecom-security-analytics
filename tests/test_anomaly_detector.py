"""
Tests for src/anomaly_detector.py
"""

import os
import json
import pytest
import numpy as np


class TestFeatureSelection:
    def test_beacon_features_list_exists(self):
        import anomaly_detector as ad
        assert isinstance(ad.BEACON_FEATURES, list)
        assert len(ad.BEACON_FEATURES) >= 4

    def test_context_features_excluded(self):
        """Regression: hour_of_day/day_of_week must NOT be in BEACON_FEATURES
        - we removed them because they polluted the anomaly score."""
        import anomaly_detector as ad
        for bad in ("hour_of_day", "day_of_week"):
            assert bad not in ad.BEACON_FEATURES, \
                f"{bad} should not be a beacon feature"


class TestScoring:
    def test_prepare_matrix_returns_clean_dataframe(self, features_table):
        import anomaly_detector as ad
        X, host = ad._prepare_matrix(features_table)
        assert X.shape[0] == len(host)
        assert X.shape[1] == len(ad.BEACON_FEATURES)
        assert not X.isnull().any().any()

    def test_prepare_matrix_population_scaling(self, features_table):
        import anomaly_detector as ad
        X, _ = ad._prepare_matrix(features_table)
        if "event_count" in X.columns:
            spread = X["event_count"].std()
            assert spread > 0.0


class TestModelArtifacts:
    def test_iso_forest_file_exists(self, project_root):
        p = os.path.join(project_root, "models", "iso_forest.pkl")
        assert os.path.exists(p)

    def test_anomaly_scores_csv_exists(self, reports_dir):
        p = os.path.join(reports_dir, "anomaly_scores.csv")
        assert os.path.exists(p)

    def test_anomaly_scores_columns(self, reports_dir):
        import pandas as pd
        p = os.path.join(reports_dir, "anomaly_scores.csv")
        if not os.path.exists(p):
            pytest.skip()
        df = pd.read_csv(p)
        for col in ("entity", "window_start", "anomaly_score",
                    "is_anomaly", "is_compromised_host"):
            assert col in df.columns


class TestMetrics:
    def test_anomaly_metrics_exists(self, reports_dir):
        p = os.path.join(reports_dir, "anomaly_metrics.json")
        assert os.path.exists(p)

    def test_precision_at_k_present(self, reports_dir):
        p = os.path.join(reports_dir, "anomaly_metrics.json")
        if not os.path.exists(p):
            pytest.skip()
        with open(p) as f:
            m = json.load(f)
        assert "precision_at_k" in m

    def test_top_k_precision_is_strong(self, reports_dir):
        """Regression: Precision@10 and Precision@25 must remain high."""
        p = os.path.join(reports_dir, "anomaly_metrics.json")
        if not os.path.exists(p):
            pytest.skip()
        with open(p) as f:
            m = json.load(f)
        pak = m["precision_at_k"]
        assert pak["top_10"]["precision"] >= 0.8
        assert pak["top_25"]["precision"] >= 0.7

    def test_roc_auc_reasonable(self, reports_dir):
        p = os.path.join(reports_dir, "anomaly_metrics.json")
        if not os.path.exists(p):
            pytest.skip()
        with open(p) as f:
            m = json.load(f)
        assert m["roc_auc"] >= 0.5
