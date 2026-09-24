"""Tests for src/experiments.py and src/response_and_brief.py"""
import os
import json
import sys
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))


class TestSimulationArtefacts:
    def test_simulation_file_exists(self):
        p = os.path.join(ROOT, "reports", "simulation_results.json")
        assert os.path.exists(p)

    def test_simulation_has_three_configurations(self):
        p = os.path.join(ROOT, "reports", "simulation_results.json")
        if not os.path.exists(p):
            pytest.skip()
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "baseline" in data
        assert "alternative_1_c2_only" in data
        assert "alternative_2_auth_only" in data

    def test_simulation_shows_partial_coverage(self):
        p = os.path.join(ROOT, "reports", "simulation_results.json")
        if not os.path.exists(p):
            pytest.skip()
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        coverage = data["comparison"]["detection_coverage"]
        # Baseline covers all three
        assert coverage["c2_beacon"]["baseline"]
        assert coverage["dns_tunnel"]["baseline"]
        assert coverage["lateral_movement"]["baseline"]
        # C2-only does not cover DNS or lateral
        assert not coverage["dns_tunnel"]["alt1"]
        assert not coverage["lateral_movement"]["alt1"]
        # Auth-only does not cover C2 or DNS
        assert not coverage["c2_beacon"]["alt2"]
        assert not coverage["dns_tunnel"]["alt2"]


class TestAdversarialArtefacts:
    def test_adversarial_file_exists(self):
        p = os.path.join(ROOT, "reports", "adversarial_results.json")
        assert os.path.exists(p)

    def test_three_scenarios_documented(self):
        p = os.path.join(ROOT, "reports", "adversarial_results.json")
        if not os.path.exists(p):
            pytest.skip()
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["n_scenarios"] == 3
        scenario_names = [s["scenario"] for s in data["scenarios"]]
        assert "jittered_beacon" in scenario_names
        assert "low_entropy_tunnel" in scenario_names
        assert "port_shifted_lateral" in scenario_names

    def test_each_scenario_has_evasion_boundary(self):
        p = os.path.join(ROOT, "reports", "adversarial_results.json")
        if not os.path.exists(p):
            pytest.skip()
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        for s in data["scenarios"]:
            assert "still_detected" in s
            assert "affected_detector" in s


class TestResponsePlaybook:
    def test_playbook_file_exists(self):
        p = os.path.join(ROOT, "reports", "response_playbook.json")
        assert os.path.exists(p)

    def test_playbook_has_plans(self):
        p = os.path.join(ROOT, "reports", "response_playbook.json")
        if not os.path.exists(p):
            pytest.skip()
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["n_plans"] > 0
        assert len(data["plans"]) == data["n_plans"]

    def test_plans_have_required_sections(self):
        p = os.path.join(ROOT, "reports", "response_playbook.json")
        if not os.path.exists(p):
            pytest.skip()
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        for plan in data["plans"][:5]:
            assert "containment" in plan
            assert "eradication" in plan
            assert "recovery" in plan
            assert len(plan["containment"]) > 0


class TestExecutiveBrief:
    def test_brief_file_exists(self):
        p = os.path.join(ROOT, "reports", "executive_brief.md")
        assert os.path.exists(p)

    def test_brief_has_required_sections(self):
        p = os.path.join(ROOT, "reports", "executive_brief.md")
        if not os.path.exists(p):
            pytest.skip()
        with open(p, "r", encoding="utf-8") as f:
            content = f.read()
        assert "# Executive Security Brief" in content
        assert "## Headline" in content
        assert "## Recommended actions" in content
        assert "correlated incidents" in content

    def test_brief_mentions_attack_techniques(self):
        p = os.path.join(ROOT, "reports", "executive_brief.md")
        if not os.path.exists(p):
            pytest.skip()
        with open(p, "r", encoding="utf-8") as f:
            content = f.read()
        assert "ATT&CK" in content or "ATTACK" in content
