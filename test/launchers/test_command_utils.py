import subprocess
import sys
import unittest
from unittest.mock import patch, MagicMock

from graphsignal.launchers.command_utils import (
    extract_host, extract_port, hash_workload_id, resolve_metrics_host,
    resolve_metrics_port, start_watcher)


class HashWorkloadIdTest(unittest.TestCase):
    def test_returns_12_char_hash(self):
        h = hash_workload_id(['vllm', 'serve', 'meta-llama/Llama-3-70B',
                              '--tensor-parallel-size', '2'])
        self.assertEqual(len(h), 12)

    def test_same_command_same_hash(self):
        args = ['sglang.launch_server', '--model-path', 'm', '--tp', '2']
        self.assertEqual(hash_workload_id(args),
                         hash_workload_id(list(args)))

    def test_flag_change_changes_hash(self):
        h1 = hash_workload_id(['vllm', 'serve', 'm', '--max-num-seqs', '256'])
        h2 = hash_workload_id(['vllm', 'serve', 'm', '--max-num-seqs', '512'])
        self.assertNotEqual(h1, h2)


class ExtractHostTest(unittest.TestCase):
    def test_space_form(self):
        self.assertEqual(extract_host(['trtllm-serve', '--host', '0.0.0.0']),
                         '0.0.0.0')

    def test_equals_form(self):
        self.assertEqual(extract_host(['trtllm-serve', '--host=localhost']),
                         'localhost')

    def test_absent_returns_default(self):
        self.assertIsNone(extract_host(['trtllm-serve']))
        self.assertEqual(extract_host(['trtllm-serve'], default='localhost'),
                         'localhost')


class ResolveMetricsHostTest(unittest.TestCase):
    def test_explicit_host_wins(self):
        self.assertEqual(
            resolve_metrics_host('10.0.0.1', ['trtllm-serve', '--host', '0.0.0.0'],
                                 default='localhost'),
            '10.0.0.1')

    def test_falls_back_to_engine_host(self):
        self.assertEqual(
            resolve_metrics_host(None, ['trtllm-serve', '--host', '0.0.0.0'],
                                 default='localhost'),
            '0.0.0.0')

    def test_falls_back_to_default(self):
        self.assertEqual(
            resolve_metrics_host(None, ['trtllm-serve'], default='localhost'),
            'localhost')


class ExtractPortTest(unittest.TestCase):
    def test_space_form(self):
        self.assertEqual(extract_port(['vllm', 'serve', '--port', '8001']), 8001)

    def test_equals_form(self):
        self.assertEqual(extract_port(['vllm', 'serve', '--port=8001']), 8001)

    def test_absent_returns_default(self):
        self.assertIsNone(extract_port(['vllm', 'serve']))
        self.assertEqual(extract_port(['vllm', 'serve'], default=8000), 8000)

    def test_non_int_returns_default(self):
        self.assertEqual(extract_port(['vllm', '--port', 'abc'], default=8000), 8000)

    def test_trailing_port_flag_without_value(self):
        self.assertEqual(extract_port(['vllm', '--port'], default=8000), 8000)


class ResolveMetricsPortTest(unittest.TestCase):
    def test_explicit_port_wins(self):
        # Explicit --metrics-port overrides both the engine --port and default.
        self.assertEqual(
            resolve_metrics_port(9999, ['vllm', '--port', '8001'], default=8000), 9999)

    def test_falls_back_to_engine_port(self):
        self.assertEqual(
            resolve_metrics_port(None, ['vllm', '--port', '8001'], default=8000), 8001)

    def test_falls_back_to_default(self):
        self.assertEqual(
            resolve_metrics_port(None, ['vllm', 'serve'], default=8000), 8000)

    def test_no_default_returns_none(self):
        self.assertIsNone(resolve_metrics_port(None, ['app.py'], default=None))


class StartWatcherTest(unittest.TestCase):
    def test_spawn_args_default(self):
        fake_popen = MagicMock(name='Popen')
        with patch.object(subprocess, 'Popen', return_value=fake_popen) as popen_m:
            result = start_watcher(12345)

        self.assertIs(result, fake_popen)
        popen_m.assert_called_once()
        cmd = popen_m.call_args[0][0]
        kwargs = popen_m.call_args[1]

        self.assertEqual(cmd[0], sys.executable)
        self.assertEqual(cmd[1:5], ['-m', 'graphsignal.commands.graphsignal_watch',
                                    '--pid', '12345'])
        self.assertNotIn('--workload-id', cmd)
        self.assertNotIn('--otel-collector-port', cmd)
        self.assertNotIn('--metrics-port', cmd)
        self.assertTrue(kwargs.get('start_new_session'))

    def test_spawn_args_with_workload_id(self):
        with patch.object(subprocess, 'Popen', return_value=MagicMock()) as popen_m:
            start_watcher(12345, workload_id='wl-abc123')
        cmd = popen_m.call_args[0][0]
        self.assertEqual(cmd[5:7], ['--workload-id', 'wl-abc123'])

    def test_spawn_args_with_otel_port(self):
        with patch.object(subprocess, 'Popen', return_value=MagicMock()) as popen_m:
            start_watcher(54321, otel_collector_port=4317)
        cmd = popen_m.call_args[0][0]
        self.assertEqual(cmd[5:], ['--otel-collector-port', '4317'])

    def test_spawn_args_with_metrics_port(self):
        with patch.object(subprocess, 'Popen', return_value=MagicMock()) as popen_m:
            start_watcher(54321, metrics_port=8000)
        cmd = popen_m.call_args[0][0]
        self.assertEqual(cmd[5:], ['--metrics-port', '8000'])

    def test_spawn_args_with_metrics_path(self):
        with patch.object(subprocess, 'Popen', return_value=MagicMock()) as popen_m:
            start_watcher(54321, metrics_port=8000,
                          metrics_path='/prometheus/metrics')
        cmd = popen_m.call_args[0][0]
        self.assertEqual(
            cmd[5:],
            ['--metrics-port', '8000', '--metrics-path', '/prometheus/metrics'])

    def test_spawn_args_with_metrics_host(self):
        with patch.object(subprocess, 'Popen', return_value=MagicMock()) as popen_m:
            start_watcher(54321, metrics_port=8000, metrics_host='localhost')
        cmd = popen_m.call_args[0][0]
        self.assertEqual(
            cmd[5:], ['--metrics-port', '8000', '--metrics-host', 'localhost'])

    def test_spawn_args_with_both_ports(self):
        with patch.object(subprocess, 'Popen', return_value=MagicMock()) as popen_m:
            start_watcher(54321, otel_collector_port=4317, metrics_port=8000)
        cmd = popen_m.call_args[0][0]
        self.assertEqual(
            cmd[5:], ['--otel-collector-port', '4317', '--metrics-port', '8000'])

    def test_spawn_args_with_workload_and_ports(self):
        with patch.object(subprocess, 'Popen', return_value=MagicMock()) as popen_m:
            start_watcher(54321, workload_id='wl-abc123',
                          otel_collector_port=4317, metrics_port=8000)
        cmd = popen_m.call_args[0][0]
        self.assertEqual(
            cmd[5:],
            ['--workload-id', 'wl-abc123',
             '--otel-collector-port', '4317', '--metrics-port', '8000'])

    def test_returns_none_on_failure(self):
        with patch.object(subprocess, 'Popen', side_effect=OSError('boom')):
            result = start_watcher(1)
        self.assertIsNone(result)


if __name__ == '__main__':
    unittest.main()
