"""
Tests for src/threat_intelligence.py
"""

import os
import json
import pytest
import pandas as pd


class TestLoaders:
    def test_ioc_list_exists(self, data_dir):
        p = os.path.join(data_dir, "threat_intelligence", "ioc_list.csv")
        assert os.path.exists(p)

    def test_pattern_library_exists(self, data_dir):
        p = os.path.join(data_dir, "threat_intelligence", "attack_patterns.csv")
        assert os.path.exists(p)

    def test_load_iocs(self, project_root):
        import threat_intelligence as ti
        p = os.path.join(project_root, "data", "threat_intelligence", "ioc_list.csv")
        iocs = ti.load_iocs(p)
        assert len(iocs) >= 5
        for col in ("ioc_type", "value", "threat_actor", "confidence"):
            assert col in iocs.columns

    def test_load_patterns(self, project_root):
        import threat_intelligence as ti
        p = os.path.join(project_root, "data", "threat_intelligence",
                         "attack_patterns.csv")
        pats = ti.load_patterns(p)
        assert len(pats) >= 4


class TestTextConversion:
    def test_incident_to_text_produces_string(self, enriched_incidents):
        import threat_intelligence as ti
        if not enriched_incidents:
            pytest.skip()
        text = ti.incident_to_text(enriched_incidents[0])
        assert isinstance(text, str)
        assert len(text) > 10


class TestScoring:
    def test_compute_ti_score_bounded(self):
        import threat_intelligence as ti
        assert ti.compute_ti_score([], []) == 0.0
        strong = [
            {"confidence": 0.95}, {"confidence": 0.90}, {"confidence": 0.85}
        ]
        assert ti.compute_ti_score(strong, []) <= 15.0

    def test_compute_ti_score_positive_with_matches(self):
        import threat_intelligence as ti
        score = ti.compute_ti_score([{"confidence": 0.9}], [])
        assert score > 0


class TestEnrichedOutput:
    def test_enriched_incidents_file_exists(self, reports_dir):
        p = os.path.join(reports_dir, "enriched_incidents.json")
        assert os.path.exists(p)

    def test_enriched_incidents_have_threat_intel_block(self, enriched_incidents):
        if not enriched_incidents:
            pytest.skip()
        for inc in enriched_incidents[:10]:
            assert "threat_intel" in inc
            assert "matched_iocs" in inc["threat_intel"]
            assert "similar_attacks" in inc["threat_intel"]
            assert "ti_score" in inc["threat_intel"]
            assert "final_score" in inc["threat_intel"]

    def test_ti_score_bounded(self, enriched_incidents):
        for inc in enriched_incidents:
            assert 0 <= inc["threat_intel"]["ti_score"] <= 15.0

    def test_top_incidents_have_ioc_matches(self, enriched_incidents):
        """Regression: the top incidents should match at least one IOC."""
        top = enriched_incidents[:5]
        with_ioc = sum(1 for inc in top
                       if inc["threat_intel"]["matched_ioc_count"] > 0)
        assert with_ioc >= 3, \
            f"Only {with_ioc}/5 top incidents matched an IOC"


class TestSummary:
    def test_ti_summary_exists(self, reports_dir):
        p = os.path.join(reports_dir, "threat_intelligence_summary.json")
        assert os.path.exists(p)

    def test_summary_structure(self, reports_dir):
        p = os.path.join(reports_dir, "threat_intelligence_summary.json")
        if not os.path.exists(p):
            pytest.skip()
        with open(p) as f:
            s = json.load(f)
        for k in ("n_incidents", "incidents_with_ioc_match",
                  "threat_actor_distribution", "max_final_score"):
            assert k in s
