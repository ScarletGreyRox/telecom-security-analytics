"""
Tests for src/data_generator.py

Verifies:
  - All three telemetry sources are generated
  - Multi-day span with diurnal pattern
  - Injected attack scenarios are present
"""

import pytest
import pandas as pd


class TestNetflowGeneration:
    def test_netflow_has_expected_columns(self, raw_netflow):
        expected = {"timestamp", "src_ip", "dst_ip", "src_port", "dst_port",
                    "bytes_sent", "bytes_recvd", "protocol", "action"}
        assert expected.issubset(set(raw_netflow.columns))

    def test_netflow_row_count(self, raw_netflow):
        # 7 days of traffic; expect tens of thousands of rows
        assert len(raw_netflow) > 30_000

    def test_netflow_timestamp_is_datetime(self, raw_netflow):
        assert pd.api.types.is_datetime64_any_dtype(raw_netflow["timestamp"])

    def test_netflow_spans_multiple_days(self, raw_netflow):
        span = raw_netflow["timestamp"].max() - raw_netflow["timestamp"].min()
        assert span.days >= 6, "Expected at least 6 days of telemetry"

    def test_action_values_are_valid(self, raw_netflow):
        assert set(raw_netflow["action"].unique()).issubset({"ALLOW", "DENY"})

    def test_protocol_values_are_valid(self, raw_netflow):
        assert set(raw_netflow["protocol"].unique()).issubset({"TCP", "UDP", "ICMP"})

    def test_bytes_are_non_negative(self, raw_netflow):
        assert (raw_netflow["bytes_sent"] >= 0).all()
        assert (raw_netflow["bytes_recvd"] >= 0).all()


class TestInjectedAttacks:
    def test_compromised_hosts_present(self, raw_netflow):
        """The three injected hosts must appear in the netflow log."""
        for host in ["10.0.1.50", "10.0.2.77", "10.0.3.111"]:
            assert (raw_netflow["src_ip"] == host).any(), \
                f"Injected host {host} not found in netflow"

    def test_beacon_host_has_high_event_rate(self, raw_netflow):
        """The beacon host should produce far more events than the median host."""
        counts = raw_netflow["src_ip"].value_counts()
        beacon_count = counts.get("10.0.1.50", 0)
        median_count = counts.median()
        assert beacon_count > 3 * median_count

    def test_exfil_host_has_large_transfers(self, raw_netflow):
        """The exfil host should have at least one huge outbound transfer."""
        exfil = raw_netflow[raw_netflow["src_ip"] == "10.0.3.111"]
        assert exfil["bytes_sent"].max() > 5_000_000

    def test_external_c2_ips_present(self, raw_netflow):
        """Netflow should include traffic to the injected C2 IPs."""
        dst_ips = set(raw_netflow["dst_ip"].unique())
        assert "172.16.99.10" in dst_ips or "172.16.99.20" in dst_ips


class TestDNSGeneration:
    def test_dns_has_expected_columns(self, raw_dns):
        expected = {"timestamp", "src_ip", "query", "query_type",
                    "response_code", "domain"}
        assert expected.issubset(set(raw_dns.columns))

    def test_dns_row_count(self, raw_dns):
        assert len(raw_dns) > 20_000

    def test_malicious_domain_present(self, raw_dns):
        """The DNS tunnel domain should appear in the DNS logs."""
        domains = set(raw_dns["domain"].str.lower().unique())
        assert "dns-tunnel.xyz" in domains

    def test_benign_domains_dominant(self, raw_dns):
        """Benign domains should dominate — malicious ones are a small minority."""
        malicious = raw_dns["domain"].isin(
            ["evil-c2.example", "dns-tunnel.xyz", "badactor-update.net",
             "malware-cdn.top", "apt-relay.cc"]
        )
        malicious_ratio = malicious.mean()
        assert 0.02 < malicious_ratio < 0.30


class TestAuthGeneration:
    def test_auth_has_expected_columns(self, raw_auth):
        expected = {"timestamp", "user", "src_ip", "host",
                    "event_type", "status", "auth_method"}
        assert expected.issubset(set(raw_auth.columns))

    def test_auth_row_count(self, raw_auth):
        assert len(raw_auth) > 5_000

    def test_failure_ratio_is_realistic(self, raw_auth):
        """Around 5-15% failures is realistic for synthetic auth logs."""
        fr = (raw_auth["status"] == "FAILURE").mean()
        assert 0.01 < fr < 0.25

    def test_injected_lateral_user_present(self, raw_auth):
        """user007 is our injected lateral-movement user."""
        assert (raw_auth["user"] == "user007").any()
