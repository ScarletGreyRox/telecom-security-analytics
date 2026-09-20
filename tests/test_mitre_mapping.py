"""
Tests for src/mitre_mapping.py
"""

import os
import json
import pytest


class TestTechniqueCatalogue:
    def test_technique_catalogue_exists(self):
        import mitre_mapping as mm
        assert isinstance(mm.TECHNIQUES, dict)
        assert len(mm.TECHNIQUES) >= 8

    def test_every_technique_has_name_and_tactic(self):
        import mitre_mapping as mm
        for tid, t in mm.TECHNIQUES.items():
            assert "name" in t and t["name"]
            assert "tactic" in t and t["tactic"]

    def test_detector_mapping_complete(self):
        import mitre_mapping as mm
        for det in ("c2_anomaly", "dns_classifier", "auth_lateral"):
            assert det in mm.DETECTOR_TECHNIQUES
            assert len(mm.DETECTOR_TECHNIQUES[det]) >= 1


class TestMappingFunction:
    def test_no_detectors_no_techniques(self):
        import mitre_mapping as mm
        assert mm.map_incident_to_techniques({"detectors": []}) == []

    def test_single_detector_produces_techniques(self):
        import mitre_mapping as mm
        techs = mm.map_incident_to_techniques({"detectors": ["c2_anomaly"]})
        assert len(techs) >= 1
        for t in techs:
            assert "id" in t and "name" in t and "tactic" in t and "confidence" in t

    def test_deduplication(self):
        import mitre_mapping as mm
        a = mm.map_incident_to_techniques({"detectors": ["c2_anomaly"]})
        b = mm.map_incident_to_techniques({"detectors": ["c2_anomaly", "c2_anomaly"]})
        assert len(a) == len(b)

    def test_multi_detector_union(self):
        import mitre_mapping as mm
        both = mm.map_incident_to_techniques(
            {"detectors": ["c2_anomaly", "auth_lateral"]})
        c2_only = mm.map_incident_to_techniques({"detectors": ["c2_anomaly"]})
        assert len(both) > len(c2_only)


class TestTaggedOutput:
    def test_tagged_incidents_file_exists(self, reports_dir):
        p = os.path.join(reports_dir, "mitre_tagged_incidents.json")
        assert os.path.exists(p)

    def test_tagged_incidents_have_mitre_fields(self, tagged_incidents):
        if not tagged_incidents:
            pytest.skip()
        for inc in tagged_incidents[:10]:
            assert "mitre_techniques" in inc
            assert "mitre_tactics" in inc
            assert "technique_count" in inc

    def test_every_incident_has_at_least_one_technique(self, tagged_incidents):
        for inc in tagged_incidents:
            assert inc["technique_count"] >= 1


class TestSummary:
    def test_mitre_summary_exists(self, reports_dir):
        p = os.path.join(reports_dir, "mitre_summary.json")
        assert os.path.exists(p)

    def test_summary_lists_all_tactics(self, reports_dir):
        p = os.path.join(reports_dir, "mitre_summary.json")
        if not os.path.exists(p):
            pytest.skip()
        with open(p) as f:
            s = json.load(f)
        tactics = set(s.get("tactic_frequency", {}).keys())
        assert "Command and Control" in tactics or "Exfiltration" in tactics
