import unittest

from graphsignal.launchers import vllm_launcher as vllm_mod
from graphsignal.launchers.command_utils import hash_workload_id
from graphsignal.launchers.vllm_launcher import VllmLauncher, _inject_vllm_args

from test.launchers._helpers import LaunchFixture


class VllmMatchTest(unittest.TestCase):
    def test_matches_vllm_command(self):
        self.assertTrue(VllmLauncher(['vllm', 'serve', 'model']).match())
        self.assertTrue(VllmLauncher(['/abs/path/vllm', 'serve']).match())

    def test_does_not_match_other_commands(self):
        self.assertFalse(VllmLauncher(['python', 'app.py']).match())
        self.assertFalse(VllmLauncher(['vllm-something', 'serve']).match())
        self.assertFalse(VllmLauncher([]).match())


class VllmArgInjectionTest(unittest.TestCase):
    def test_appends_otlp_endpoint_when_missing(self):
        out = _inject_vllm_args(['vllm', 'serve', 'm'], otel_port=4317)
        self.assertEqual(out, ['vllm', 'serve', 'm', '--otlp-traces-endpoint', '127.0.0.1:4317'])

    def test_no_otel_port_means_no_endpoint_injection(self):
        # When the caller passes otel_port=None (user already supplied an
        # endpoint), the helper must not append anything endpoint-related.
        out = _inject_vllm_args(
            ['vllm', 'serve', '--otlp-traces-endpoint', '127.0.0.1:9999'], otel_port=None)
        self.assertEqual(out, ['vllm', 'serve', '--otlp-traces-endpoint', '127.0.0.1:9999'])

    def test_preserves_user_otlp_endpoint(self):
        out = _inject_vllm_args(
            ['vllm', 'serve', '--otlp-traces-endpoint', '127.0.0.1:9999'], otel_port=4317)
        self.assertEqual(out, ['vllm', 'serve', '--otlp-traces-endpoint', '127.0.0.1:9999'])

    def test_preserves_user_otlp_endpoint_equals_form(self):
        out = _inject_vllm_args(
            ['vllm', 'serve', '--otlp-traces-endpoint=127.0.0.1:9999'], otel_port=4317)
        self.assertEqual(out, ['vllm', 'serve', '--otlp-traces-endpoint=127.0.0.1:9999'])

    def test_strips_disable_log_stats(self):
        # Stripping --disable-log-stats applies even when we're not injecting
        # an endpoint (user is exporting traces elsewhere but we still want
        # vLLM emitting metrics).
        out = _inject_vllm_args(
            ['vllm', 'serve', 'm', '--otlp-traces-endpoint', 'http://them:4317',
             '--disable-log-stats'], otel_port=None)
        self.assertNotIn('--disable-log-stats', out)


class VllmLaunchTest(unittest.TestCase):
    def test_launch_calls_pipeline_in_order_when_otel_enabled(self):
        launcher = VllmLauncher(['vllm', 'serve', 'm'], enable_otel=True)
        with LaunchFixture(vllm_mod) as fx:
            launcher.launch()

        fx.setup_env_m.assert_called_once()
        fx.find_port_m.assert_called_once()
        # No --port in argv → falls back to vLLM's default serving port (8000).
        fx.launch_supervised_m.assert_called_once_with(
            ['/abs/exec', 'serve', 'm', '--otlp-traces-endpoint', '127.0.0.1:4242'],
            workload_id=hash_workload_id(['vllm', 'serve', 'm']),
            otel_collector_port=4242, metrics_port=8000)

    def test_launch_no_otel_by_default(self):
        # Default (no --enable-otel): no collector port, no endpoint injection.
        launcher = VllmLauncher(['vllm', 'serve', 'm'])
        with LaunchFixture(vllm_mod) as fx:
            launcher.launch()

        fx.find_port_m.assert_not_called()
        fx.launch_supervised_m.assert_called_once_with(
            ['/abs/exec', 'serve', 'm'],
            workload_id=hash_workload_id(['vllm', 'serve', 'm']),
            otel_collector_port=None, metrics_port=8000)
        self.assertNotIn('--otlp-traces-endpoint', fx.launched_argv)

    def test_launch_metrics_port_from_engine_args(self):
        launcher = VllmLauncher(['vllm', 'serve', 'm', '--port', '8001'])
        with LaunchFixture(vllm_mod) as fx:
            launcher.launch()
        fx.launch_supervised_m.assert_called_once_with(
            ['/abs/exec', 'serve', 'm', '--port', '8001'],
            workload_id=hash_workload_id(['vllm', 'serve', 'm', '--port', '8001']),
            otel_collector_port=None, metrics_port=8001)

    def test_launch_explicit_metrics_port_overrides_engine_args(self):
        launcher = VllmLauncher(
            ['vllm', 'serve', 'm', '--port', '8001'], metrics_port=9999)
        with LaunchFixture(vllm_mod) as fx:
            launcher.launch()
        fx.launch_supervised_m.assert_called_once_with(
            ['/abs/exec', 'serve', 'm', '--port', '8001'],
            workload_id=hash_workload_id(['vllm', 'serve', 'm', '--port', '8001']),
            otel_collector_port=None, metrics_port=9999)

    def test_launch_skips_collector_when_user_endpoint_present(self):
        # enable_otel + user-supplied --otlp-traces-endpoint → no find_port,
        # no collector in the watcher, argv passed through unchanged.
        launcher = VllmLauncher(
            ['vllm', 'serve', 'm', '--otlp-traces-endpoint', 'http://them:4317'],
            enable_otel=True)
        with LaunchFixture(vllm_mod) as fx:
            launcher.launch()

        fx.find_port_m.assert_not_called()
        fx.launch_supervised_m.assert_called_once_with(
            ['/abs/exec', 'serve', 'm', '--otlp-traces-endpoint', 'http://them:4317'],
            workload_id=hash_workload_id(
                ['vllm', 'serve', 'm', '--otlp-traces-endpoint', 'http://them:4317']),
            otel_collector_port=None, metrics_port=8000)

    def test_launch_raises_when_executable_missing(self):
        launcher = VllmLauncher(['vllm-typo'])
        with LaunchFixture(vllm_mod) as fx:
            fx.resolve_m.return_value = None
            with self.assertRaises(FileNotFoundError):
                launcher.launch()
        fx.launch_supervised_m.assert_not_called()


if __name__ == '__main__':
    unittest.main()
