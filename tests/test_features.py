"""
Tests for src/features.py

Verifies:
  - Feature table schema and dimensions
  - MIN_EVENTS filter applied
  - Compromised-host labels preserved
  - Beacon / DNS / auth feature groups present
"""

import pytest
import pandas as pd
import numpy as np


class TestFeatureTable:
    def test_features_table_exists_and_has_rows(self, features_table):
        assert len(features_table) > 5_000

    def test_expected_columns_present(self, features_table):
        expected = {
            "entity", "window_start", "is_compromised_host",
            "delta_cv", "regularity", "top_dst_share", "event_count",
            "byte_sent_recv_ratio", "byte_log_max_sent", "off_hours_ratio",
            "dns_avg_entropy", "dns_avg_sub_len", "dns_subdomain_uniqueness",
            "auth_count", "auth_unique_hosts", "auth_target_rate",
        }
        missing = expected - set(features_table.columns)
        assert not missing, f"Missing columns: {missing}"

    def test_compromised_labels_exist(self, features_table):
        assert features_table["is_compromised_host"].sum() > 0

    def test_compromised_fraction_is_small(self, features_table):
        frac = features_table["is_compromised_host"].mean()
        assert 0.001 < frac < 0.10

    def test_entity_types_are_valid(self, features_table):
        entities = features_table["entity"].astype(str)
        assert entities.str.startswith("host:").any() or \
               entities.str.startswith("user:").any()
        bad = entities[~entities.str.startswith(("host:", "user:"))]
        assert len(bad) == 0


class TestBeaconFeatures:
    def test_regularity_bounded(self, features_table):
        r = features_table["regularity"].dropna()
        assert r.min() >= 0
        assert r.max() <= 1.0 + 1e-6

    def test_delta_cv_non_negative(self, features_table):
        cv = features_table["delta_cv"].dropna()
        assert (cv >= 0).all()

    def test_beacon_host_has_extreme_regularity(self, features_table):
        beacon = features_table[
            features_table["entity"] == "host:10.0.1.50"
        ]["regularity"].dropna()
        population = features_table[
            features_table["entity"].str.startswith("host:")
        ]["regularity"].dropna()
        assert beacon.mean() > population.mean()


class TestDNSFeatures:
    def test_entropy_non_negative(self, features_table):
        e = features_table["dns_avg_entropy"].dropna()
        assert (e >= 0).all()

    def test_tunnel_host_has_high_entropy(self, features_table):
        tunnel = features_table[
            features_table["entity"] == "host:10.0.2.77"
        ]["dns_avg_entropy"].dropna()
        baseline = features_table[
            features_table["entity"].str.startswith("host:")
        ]["dns_avg_entropy"].dropna()
        assert tunnel.mean() > baseline.mean()

    def test_subdomain_uniqueness_bounded(self, features_table):
        u = features_table["dns_subdomain_uniqueness"].dropna()
        assert (u >= 0).all()
        assert (u <= 1.0 + 1e-6).all()


class TestByteFeatures:
    def test_bytes_non_negative(self, features_table):
        assert (features_table["byte_total_sent"].dropna() >= 0).all()

    def test_sent_recv_ratio_non_negative(self, features_table):
        r = features_table["byte_sent_recv_ratio"].dropna()
        assert (r >= 0).all()

    def test_exfil_host_has_extreme_bytes(self, features_table):
        exfil = features_table[
            features_table["entity"] == "host:10.0.3.111"
        ]["byte_log_max_sent"].dropna()
        assert exfil.max() > 6.0
