import unittest

from graphsignal.launchers import sglang_launcher as sglang_mod
from graphsignal.launchers.command_utils import hash_workload_id
from graphsignal.launchers.sglang_launcher import SglangLauncher, _inject_sglang_args

from test.launchers._helpers import LaunchFixture


class SglangMatchTest(unittest.TestCase):
    def test_matches_sglang_executable(self):
        self.assertTrue(SglangLauncher(['sglang', 'serve']).match())
        self.assertTrue(SglangLauncher(['sglang.launch_server', '--model', 'foo']).match())
        self.assertTrue(SglangLauncher(['/usr/bin/sglang']).match())

    def test_matches_python_m_form(self):
        self.assertTrue(
            SglangLauncher(['python', '-m', 'sglang.launch_server', '--model', 'foo']).match())
        self.assertTrue(
            SglangLauncher(['python3.11', '-m', 'sglang.launch_server']).match())
        self.assertTrue(
            SglangLauncher(['/usr/bin/python', '-m', 'sglang.launch_server']).match())

    def test_does_not_match(self):
        self.assertFalse(SglangLauncher(['python', 'app.py']).match())
        self.assertFalse(SglangLauncher(['python', '-m', 'something_else']).match())
        self.assertFalse(SglangLauncher(['sglang-other', 'serve']).match())
        self.assertFalse(SglangLauncher([]).match())


class SglangArgInjectionTest(unittest.TestCase):
    def test_appends_all_flags_when_otel_enabled(self):
        out = _inject_sglang_args(['sglang', 'serve'], otel_port=4317, enable_otel=True)
        self.assertIn('--enable-metrics', out)
        self.assertIn('--enable-trace', out)
        self.assertEqual(out[-2:], ['--otlp-traces-endpoint', '127.0.0.1:4317'])

    def test_otel_disabled_by_default(self):
        # Without enable_otel, metrics are still added but no tracing flags.
        out = _inject_sglang_args(['sglang', 'serve'], otel_port=None, enable_otel=False)
        self.assertIn('--enable-metrics', out)
        self.assertNotIn('--enable-trace', out)
        self.assertNotIn('--otlp-traces-endpoint', out)

    def test_no_otel_port_means_no_endpoint_injection(self):
        # otel_port=None (user already supplied an endpoint) → no endpoint
        # appended, but --enable-metrics / --enable-trace are still added.
        out = _inject_sglang_args(
            ['sglang', '--otlp-traces-endpoint', '127.0.0.1:9999'],
            otel_port=None, enable_otel=True)
        self.assertEqual(out.count('--otlp-traces-endpoint'), 1)
        idx = out.index('--otlp-traces-endpoint')
        self.assertEqual(out[idx + 1], '127.0.0.1:9999')
        self.assertIn('--enable-metrics', out)
        self.assertIn('--enable-trace', out)

    def test_preserves_existing_enable_flags(self):
        out = _inject_sglang_args(
            ['sglang', 'serve', '--enable-metrics', '--enable-trace'],
            otel_port=4317, enable_otel=True)
        self.assertEqual(out.count('--enable-metrics'), 1)
        self.assertEqual(out.count('--enable-trace'), 1)

    def test_keeps_user_custom_endpoint(self):
        # Even when otel_port is supplied, an existing user endpoint must
        # be preserved (the launcher would normally pass None here, but the
        # helper still has to refuse to overwrite as a safety net).
        out = _inject_sglang_args(
            ['sglang', '--otlp-traces-endpoint', '127.0.0.1:9999'],
            otel_port=4317, enable_otel=True)
        idx = out.index('--otlp-traces-endpoint')
        self.assertEqual(out[idx + 1], '127.0.0.1:9999')


class SglangLaunchTest(unittest.TestCase):
    def test_launch_injects_flags_and_execs_when_otel_enabled(self):
        launcher = SglangLauncher(['sglang', 'serve'], enable_otel=True)
        with LaunchFixture(sglang_mod) as fx:
            launcher.launch()
        called_argv = fx.launched_argv
        self.assertIn('--enable-metrics', called_argv)
        self.assertIn('--enable-trace', called_argv)
        self.assertIn('--otlp-traces-endpoint', called_argv)
        self.assertIn('127.0.0.1:4242', called_argv)

    def test_launch_no_otel_by_default(self):
        # Default (no --enable-otel): no collector port, no tracing flags;
        # metrics still enabled.
        launcher = SglangLauncher(['sglang', 'serve'])
        with LaunchFixture(sglang_mod) as fx:
            launcher.launch()

        fx.find_port_m.assert_not_called()
        # No --port in argv → falls back to SGLang's default serving port (30000).
        fx.launch_supervised_m.assert_called_once_with(
            ['/abs/exec', 'serve', '--enable-metrics'],
            workload_id=hash_workload_id(['sglang', 'serve']),
            otel_collector_port=None, metrics_port=30000)
        called_argv = fx.launched_argv
        self.assertIn('--enable-metrics', called_argv)
        self.assertNotIn('--enable-trace', called_argv)
        self.assertNotIn('--otlp-traces-endpoint', called_argv)

    def test_launch_metrics_port_from_engine_args(self):
        launcher = SglangLauncher(['sglang', 'serve', '--port', '30001'])
        with LaunchFixture(sglang_mod) as fx:
            launcher.launch()
        fx.launch_supervised_m.assert_called_once_with(
            ['/abs/exec', 'serve', '--port', '30001', '--enable-metrics'],
            workload_id=hash_workload_id(['sglang', 'serve', '--port', '30001']),
            otel_collector_port=None, metrics_port=30001)

    def test_launch_explicit_metrics_port_overrides_engine_args(self):
        launcher = SglangLauncher(
            ['sglang', 'serve', '--port', '30001'], metrics_port=9999)
        with LaunchFixture(sglang_mod) as fx:
            launcher.launch()
        fx.launch_supervised_m.assert_called_once_with(
            ['/abs/exec', 'serve', '--port', '30001', '--enable-metrics'],
            workload_id=hash_workload_id(['sglang', 'serve', '--port', '30001']),
            otel_collector_port=None, metrics_port=9999)

    def test_launch_skips_collector_when_user_endpoint_present(self):
        # enable_otel + user-supplied --otlp-traces-endpoint → no find_port,
        # no collector in the watcher. --enable-metrics / --enable-trace added.
        launcher = SglangLauncher(
            ['sglang', 'serve', '--otlp-traces-endpoint', '127.0.0.1:9999'],
            enable_otel=True)
        with LaunchFixture(sglang_mod) as fx:
            launcher.launch()

        fx.find_port_m.assert_not_called()
        self.assertEqual(
            fx.launch_supervised_m.call_args.kwargs,
            {'workload_id': hash_workload_id(
                ['sglang', 'serve', '--otlp-traces-endpoint', '127.0.0.1:9999']),
             'otel_collector_port': None, 'metrics_port': 30000})
        called_argv = fx.launched_argv
        self.assertIn('--otlp-traces-endpoint', called_argv)
        idx = called_argv.index('--otlp-traces-endpoint')
        self.assertEqual(called_argv[idx + 1], '127.0.0.1:9999')
        self.assertIn('--enable-metrics', called_argv)
        self.assertIn('--enable-trace', called_argv)


if __name__ == '__main__':
    unittest.main()
