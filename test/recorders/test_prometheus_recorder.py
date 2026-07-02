import unittest
from unittest.mock import MagicMock, patch

from graphsignal.recorders.prometheus_recorder import (
    PrometheusRecorder, build_metrics_endpoint, format_metrics_host,
    normalize_metrics_path)

_PROM_BODY = """# HELP vllm_num_requests_running Number of running requests.
# TYPE vllm_num_requests_running gauge
vllm_num_requests_running 3.0
"""


class PrometheusRecorderTest(unittest.TestCase):
    def test_normalize_metrics_path(self):
        self.assertEqual(normalize_metrics_path(None), '/metrics')
        self.assertEqual(normalize_metrics_path('/prometheus/metrics'),
                         '/prometheus/metrics')
        self.assertEqual(normalize_metrics_path('prometheus/metrics'),
                         '/prometheus/metrics')

    def test_build_metrics_endpoint_localhost(self):
        self.assertEqual(
            build_metrics_endpoint(8000, metrics_path='/prometheus/metrics',
                                   metrics_host='localhost'),
            'http://localhost:8000/prometheus/metrics')

    def test_format_metrics_host_ipv6(self):
        self.assertEqual(format_metrics_host('::1'), '[::1]')

    def test_no_metrics_port_never_fetches(self):
        # Without a configured port the recorder is inert: no endpoint, no HTTP.
        recorder = PrometheusRecorder(pid=123, metrics_port=None)
        self.assertIsNone(recorder._endpoint)
        with patch.object(recorder, '_fetch_metrics') as fetch_m:
            recorder.on_tick()
        fetch_m.assert_not_called()

    def test_fetches_only_configured_port(self):
        # The only URL ever requested is the explicitly configured port; the
        # recorder must not enumerate or probe any other ports.
        recorder = PrometheusRecorder(pid=123, metrics_port=8000)
        self.assertEqual(recorder._endpoint, 'http://127.0.0.1:8000/metrics')
        with patch.object(recorder, '_fetch_metrics', return_value=_PROM_BODY) as fetch_m, \
             patch('graphsignal.sdk.sdk', return_value=MagicMock()):
            recorder.on_tick()
        fetch_m.assert_called_once_with('http://127.0.0.1:8000/metrics')

    def test_fetches_configured_path(self):
        recorder = PrometheusRecorder(
            pid=123, metrics_port=8000, metrics_path='/prometheus/metrics')
        self.assertEqual(
            recorder._endpoint, 'http://127.0.0.1:8000/prometheus/metrics')
        with patch.object(recorder, '_fetch_metrics', return_value=_PROM_BODY) as fetch_m, \
             patch('graphsignal.sdk.sdk', return_value=MagicMock()):
            recorder.on_tick()
        fetch_m.assert_called_once_with(
            'http://127.0.0.1:8000/prometheus/metrics')

    def test_emits_parsed_gauge(self):
        recorder = PrometheusRecorder(pid=123, metrics_port=8000)
        fake_sdk = MagicMock()
        with patch.object(recorder, '_fetch_metrics', return_value=_PROM_BODY), \
             patch('graphsignal.sdk.sdk', return_value=fake_sdk):
            recorder.on_tick()
        fake_sdk.set_gauge.assert_called_once()
        self.assertEqual(
            fake_sdk.set_gauge.call_args.kwargs['name'], 'vllm_num_requests_running')
        self.assertEqual(fake_sdk.set_gauge.call_args.kwargs['value'], 3.0)

    def test_skips_gauge_histogram_bucket_labels(self):
        body = """# HELP sglang:routing_key_running_req_count Distribution of routing keys.
# TYPE sglang:routing_key_running_req_count gauge
sglang:routing_key_running_req_count{gt="0",le="1",model_name="m"} 2.0
sglang:routing_key_running_req_count{gt="1",le="2",model_name="m"} 1.0
# HELP sglang:gen_throughput The generation throughput (token/s).
# TYPE sglang:gen_throughput gauge
sglang:gen_throughput{model_name="m"} 12.5
"""
        recorder = PrometheusRecorder(pid=123, metrics_port=8000)
        fake_sdk = MagicMock()
        with patch.object(recorder, '_fetch_metrics', return_value=body), \
             patch('graphsignal.sdk.sdk', return_value=fake_sdk):
            recorder.on_tick()
        fake_sdk.set_gauge.assert_called_once()
        self.assertEqual(fake_sdk.set_gauge.call_args.kwargs['name'], 'sglang:gen_throughput')
        self.assertNotIn('gt', fake_sdk.set_gauge.call_args.kwargs.get('tags', {}))

    def test_strips_pid_label(self):
        body = """# HELP sglang:num_running_reqs The number of running requests.
# TYPE sglang:num_running_reqs gauge
sglang:num_running_reqs{pid="1973",model_name="m"} 4.0
"""
        recorder = PrometheusRecorder(pid=123, metrics_port=8000)
        fake_sdk = MagicMock()
        with patch.object(recorder, '_fetch_metrics', return_value=body), \
             patch('graphsignal.sdk.sdk', return_value=fake_sdk):
            recorder.on_tick()
        tags = fake_sdk.set_gauge.call_args.kwargs['tags']
        self.assertNotIn('pid', tags)
        self.assertEqual(tags.get('model_name'), 'm')

    def test_skips_non_finite_gauge_values(self):
        body = """# HELP sglang:fwd_occupancy Forward pass GPU occupancy percentage.
# TYPE sglang:fwd_occupancy gauge
sglang:fwd_occupancy NaN
# HELP sglang:gen_throughput The generation throughput (token/s).
# TYPE sglang:gen_throughput gauge
sglang:gen_throughput 12.5
"""
        recorder = PrometheusRecorder(pid=123, metrics_port=8000)
        fake_sdk = MagicMock()
        with patch.object(recorder, '_fetch_metrics', return_value=body), \
             patch('graphsignal.sdk.sdk', return_value=fake_sdk):
            recorder.on_tick()
        fake_sdk.set_gauge.assert_called_once()
        self.assertEqual(fake_sdk.set_gauge.call_args.kwargs['name'], 'sglang:gen_throughput')
        self.assertEqual(fake_sdk.set_gauge.call_args.kwargs['value'], 12.5)

    def test_non_prometheus_body_is_not_emitted(self):
        # A wrong port that answers HTTP with non-Prometheus content must not be
        # treated as metrics (stays unverified, nothing parsed).
        recorder = PrometheusRecorder(pid=123, metrics_port=8000)
        with patch.object(recorder, '_fetch_metrics', return_value='<html>nope</html>'), \
             patch.object(recorder, '_parse_and_emit') as parse_m:
            recorder.on_tick()
        parse_m.assert_not_called()
        self.assertFalse(recorder._verified)

    def test_fetch_failure_backs_off_without_raising(self):
        recorder = PrometheusRecorder(pid=123, metrics_port=8000)
        with patch.object(recorder, '_fetch_metrics', side_effect=OSError('refused')):
            recorder.on_tick()  # must not raise
        self.assertFalse(recorder._verified)


if __name__ == '__main__':
    unittest.main()
