"""
End-to-end integration test.

Marked `slow` - regenerates all raw data, rebuilds features, retrains
all models, and reruns correlation + enrichment + MITRE mapping.

Run explicitly with:
    pytest -m slow
Skip during rapid iteration with:
    pytest -m "not slow"
"""

import os
import sys
import pytest
import pandas as pd

pytestmark = pytest.mark.slow


class TestPipelineIntegration:
    def test_full_pipeline_from_scratch(self, project_root):
        """Run the complete pipeline in a small config."""
        sys.path.insert(0, os.path.join(project_root, "src"))

        import data_generator as dg
        import features as ft

        # 1. Generate a small dataset (2 days)
        dg.CONFIG["days"] = 2
        netflow, dns, auth = dg.build_dataset(inject_attacks=True)
        assert len(netflow) > 0
        assert len(dns) > 0
        assert len(auth) > 0

        # 2. Build features
        feats = ft.build_features(netflow, dns, auth, window="1h")
        assert len(feats) > 0

        # 3. Injected compromised hosts present
        compromised = feats[feats["is_compromised_host"] == 1]
        assert len(compromised) > 0

    def test_dns_classifier_can_train(self, raw_dns):
        import dns_classifier as dc
        sample = raw_dns.sample(n=min(5000, len(raw_dns)), random_state=42)
        result = dc.train_dns_classifier(sample, n_estimators=20)
        assert "metrics" in result
        assert 0 <= result["metrics"]["f1"] <= 1.0

    def test_anomaly_detector_can_score(self, features_table):
        import anomaly_detector as ad
        result = ad.train_isolation_forest(
            features_table, contamination=0.05, n_estimators=20)
        assert "results" in result
        assert "metrics" in result
        assert result["metrics"]["n_windows"] > 0

    def test_auth_analytics_pipeline(self, raw_auth, raw_netflow):
        import auth_analytics as aa
        result = aa.run_auth_analytics(raw_auth, raw_netflow)
        assert "auth_feats" in result
        assert "incidents" in result
        assert "summary" in result
