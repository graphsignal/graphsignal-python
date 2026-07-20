import unittest
from unittest.mock import patch

from graphsignal.launchers.command_utils import hash_workload_id
from graphsignal.launchers.trtllm_launcher import TrtllmLauncher


class TrtllmMatchTest(unittest.TestCase):
    def test_matches(self):
        self.assertTrue(TrtllmLauncher(['trtllm', 'serve']).match())
        self.assertTrue(TrtllmLauncher(['trtllm-serve', '--model', 'm']).match())
        self.assertTrue(TrtllmLauncher(['trtllm-llmapi-launch']).match())
        self.assertTrue(TrtllmLauncher(['/usr/bin/trtllm-serve']).match())

    def test_does_not_match(self):
        self.assertFalse(TrtllmLauncher(['python', 'app.py']).match())
        self.assertFalse(TrtllmLauncher(['trtllm-other']).match())


class TrtllmLaunchTest(unittest.TestCase):
    def _launch(self, launcher):
        with patch('graphsignal.launchers.trtllm_launcher.CuptiProfiler.setup_env_vars', return_value=True), \
             patch('graphsignal.launchers.trtllm_launcher._resolve', return_value='/abs/trtllm-serve'), \
             patch('graphsignal.launchers.trtllm_launcher.launch_supervised') as launch_m:
            launcher.launch()
        return launch_m

    def test_argv_unchanged(self):
        launcher = TrtllmLauncher(
            ['trtllm-serve', '--model', 'm', '--port', '8000'])
        launch_m = self._launch(launcher)

        launch_m.assert_called_once_with(
            ['/abs/trtllm-serve', '--model', 'm', '--port', '8000'],
            workload_id=hash_workload_id(
                ['trtllm-serve', '--model', 'm', '--port', '8000']),
            otel_collector_port=None, metrics_port=8000,
            metrics_path='/prometheus/metrics', metrics_host='localhost')

    def test_enable_otel_flag_ignored(self):
        launcher = TrtllmLauncher(
            ['trtllm-serve', '--model', 'm'], enable_otel=True)
        launch_m = self._launch(launcher)

        launch_m.assert_called_once_with(
            ['/abs/trtllm-serve', '--model', 'm'],
            workload_id=hash_workload_id(['trtllm-serve', '--model', 'm']),
            otel_collector_port=None, metrics_port=8000,
            metrics_path='/prometheus/metrics', metrics_host='localhost')

    def test_metrics_port_from_engine_args(self):
        launcher = TrtllmLauncher(['trtllm-serve', 'm', '--port', '8001'])
        launch_m = self._launch(launcher)
        launch_m.assert_called_once_with(
            ['/abs/trtllm-serve', 'm', '--port', '8001'],
            workload_id=hash_workload_id(['trtllm-serve', 'm', '--port', '8001']),
            otel_collector_port=None, metrics_port=8001,
            metrics_path='/prometheus/metrics', metrics_host='localhost')

    def test_metrics_host_from_engine_args(self):
        launcher = TrtllmLauncher(
            ['trtllm-serve', 'm', '--host', '0.0.0.0', '--port', '8001'])
        launch_m = self._launch(launcher)
        launch_m.assert_called_once_with(
            ['/abs/trtllm-serve', 'm', '--host', '0.0.0.0', '--port', '8001'],
            workload_id=hash_workload_id(
                ['trtllm-serve', 'm', '--host', '0.0.0.0', '--port', '8001']),
            otel_collector_port=None, metrics_port=8001,
            metrics_path='/prometheus/metrics', metrics_host='0.0.0.0')

    def test_explicit_metrics_port_overrides_engine_args(self):
        launcher = TrtllmLauncher(
            ['trtllm-serve', 'm', '--port', '8001'], metrics_port=9999)
        launch_m = self._launch(launcher)
        launch_m.assert_called_once_with(
            ['/abs/trtllm-serve', 'm', '--port', '8001'],
            workload_id=hash_workload_id(['trtllm-serve', 'm', '--port', '8001']),
            otel_collector_port=None, metrics_port=9999,
            metrics_path='/prometheus/metrics', metrics_host='localhost')


if __name__ == '__main__':
    unittest.main()
