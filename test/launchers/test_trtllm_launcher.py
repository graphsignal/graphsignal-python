import os
import unittest
from unittest.mock import MagicMock, patch

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
             patch('graphsignal.launchers.trtllm_launcher.start_watcher', return_value=MagicMock()) as start_watcher_m, \
             patch('graphsignal.launchers.trtllm_launcher._resolve', return_value='/abs/trtllm-serve'), \
             patch('os.execv') as execv_m:
            launcher.launch()
        return start_watcher_m, execv_m

    def test_argv_unchanged(self):
        launcher = TrtllmLauncher(
            ['trtllm-serve', '--model', 'm', '--port', '8000'])
        start_watcher_m, execv_m = self._launch(launcher)

        start_watcher_m.assert_called_once_with(
            os.getpid(), workload_id=hash_workload_id(
                ['trtllm-serve', '--model', 'm', '--port', '8000']),
            otel_collector_port=None, metrics_port=8000,
            metrics_path='/prometheus/metrics', metrics_host='localhost')
        called_argv = execv_m.call_args[0][1]
        self.assertEqual(
            called_argv, ['trtllm-serve', '--model', 'm', '--port', '8000'])

    def test_enable_otel_flag_ignored(self):
        launcher = TrtllmLauncher(
            ['trtllm-serve', '--model', 'm'], enable_otel=True)
        start_watcher_m, execv_m = self._launch(launcher)

        start_watcher_m.assert_called_once_with(
            os.getpid(), workload_id=hash_workload_id(['trtllm-serve', '--model', 'm']),
            otel_collector_port=None, metrics_port=8000,
            metrics_path='/prometheus/metrics', metrics_host='localhost')
        called_argv = execv_m.call_args[0][1]
        self.assertEqual(called_argv, ['trtllm-serve', '--model', 'm'])

    def test_metrics_port_from_engine_args(self):
        launcher = TrtllmLauncher(['trtllm-serve', 'm', '--port', '8001'])
        start_watcher_m, _ = self._launch(launcher)
        start_watcher_m.assert_called_once_with(
            os.getpid(), workload_id=hash_workload_id(['trtllm-serve', 'm', '--port', '8001']),
            otel_collector_port=None, metrics_port=8001,
            metrics_path='/prometheus/metrics', metrics_host='localhost')

    def test_metrics_host_from_engine_args(self):
        launcher = TrtllmLauncher(
            ['trtllm-serve', 'm', '--host', '0.0.0.0', '--port', '8001'])
        start_watcher_m, _ = self._launch(launcher)
        start_watcher_m.assert_called_once_with(
            os.getpid(),
            workload_id=hash_workload_id(
                ['trtllm-serve', 'm', '--host', '0.0.0.0', '--port', '8001']),
            otel_collector_port=None, metrics_port=8001,
            metrics_path='/prometheus/metrics', metrics_host='0.0.0.0')

    def test_explicit_metrics_port_overrides_engine_args(self):
        launcher = TrtllmLauncher(
            ['trtllm-serve', 'm', '--port', '8001'], metrics_port=9999)
        start_watcher_m, _ = self._launch(launcher)
        start_watcher_m.assert_called_once_with(
            os.getpid(), workload_id=hash_workload_id(['trtllm-serve', 'm', '--port', '8001']),
            otel_collector_port=None, metrics_port=9999,
            metrics_path='/prometheus/metrics', metrics_host='localhost')


if __name__ == '__main__':
    unittest.main()
