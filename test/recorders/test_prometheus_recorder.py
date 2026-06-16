import unittest
from unittest.mock import MagicMock, patch

from graphsignal.recorders.prometheus_recorder import PrometheusRecorder

_PROM_BODY = """# HELP vllm_num_requests_running Number of running requests.
# TYPE vllm_num_requests_running gauge
vllm_num_requests_running 3.0
"""


class PrometheusRecorderTest(unittest.TestCase):
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
