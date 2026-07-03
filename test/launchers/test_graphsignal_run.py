import sys
import os
import unittest
from unittest.mock import patch

from graphsignal.commands import graphsignal_run
from graphsignal.launchers.vllm_launcher import VllmLauncher
from graphsignal.launchers.sglang_launcher import SglangLauncher
from graphsignal.launchers.trtllm_launcher import TrtllmLauncher
from graphsignal.launchers.fallback_launcher import FallbackLauncher


class GraphsignalRunDispatchTest(unittest.TestCase):
    """`graphsignal-run` walks the launcher list and dispatches to the first
    one whose match() returns True."""

    def _run_with_argv(self, argv):
        with patch.object(VllmLauncher, 'launch') as v, \
             patch.object(SglangLauncher, 'launch') as s, \
             patch.object(TrtllmLauncher, 'launch') as t, \
             patch.object(FallbackLauncher, 'launch') as f, \
             patch.object(sys, 'argv', argv):
            try:
                graphsignal_run.main()
            except SystemExit:
                pass
        return v, s, t, f

    def test_no_args_exits(self):
        with patch.object(sys, 'argv', ['graphsignal-run']):
            with self.assertRaises(SystemExit) as cm:
                graphsignal_run.main()
        self.assertEqual(cm.exception.code, 1)

    def test_vllm_wins_over_fallback(self):
        v, s, t, f = self._run_with_argv(['graphsignal-run', 'vllm', 'serve', 'm'])
        v.assert_called_once()
        s.assert_not_called()
        t.assert_not_called()
        f.assert_not_called()

    def test_sglang_executable_wins(self):
        v, s, t, f = self._run_with_argv(
            ['graphsignal-run', 'sglang', 'serve', '--model', 'm'])
        s.assert_called_once()
        v.assert_not_called()
        t.assert_not_called()
        f.assert_not_called()

    def test_sglang_python_module_form_wins(self):
        v, s, t, f = self._run_with_argv(
            ['graphsignal-run', 'python', '-m', 'sglang.launch_server'])
        s.assert_called_once()
        f.assert_not_called()

    def test_trtllm_wins(self):
        v, s, t, f = self._run_with_argv(['graphsignal-run', 'trtllm-serve', '--model', 'm'])
        t.assert_called_once()
        v.assert_not_called()
        s.assert_not_called()
        f.assert_not_called()

    def test_unrecognised_falls_back(self):
        v, s, t, f = self._run_with_argv(['graphsignal-run', 'python', 'my_app.py'])
        f.assert_called_once()
        v.assert_not_called()
        s.assert_not_called()
        t.assert_not_called()

    def test_enable_otel_flag_stripped_and_command_still_matches(self):
        # `--enable-otel` is graphsignal-run's own flag: consumed before the
        # command, so the workload command still matches its launcher.
        v, s, t, f = self._run_with_argv(
            ['graphsignal-run', '--enable-otel', 'vllm', 'serve', 'm'])
        v.assert_called_once()
        s.assert_not_called()
        t.assert_not_called()
        f.assert_not_called()

    def test_auto_flags_flag_stripped_and_command_still_matches(self):
        v, s, t, f = self._run_with_argv(
            ['graphsignal-run', '--auto-flags', 'vllm', 'serve', 'm'])
        v.assert_called_once()
        s.assert_not_called()
        t.assert_not_called()
        f.assert_not_called()


class ExtractFlagsTest(unittest.TestCase):
    def test_extracts_leading_enable_otel(self):
        self.assertEqual(
            graphsignal_run._extract_graphsignal_flags(['--enable-otel', 'vllm', 'serve']),
            (True, None, None, False, ['vllm', 'serve']))

    def test_absent_flag_defaults_off(self):
        self.assertEqual(
            graphsignal_run._extract_graphsignal_flags(['vllm', 'serve']),
            (False, None, None, False, ['vllm', 'serve']))

    def test_flag_after_command_left_for_workload(self):
        # Only leading flags are parsed; an identically named workload flag
        # later in argv is forwarded untouched.
        self.assertEqual(
            graphsignal_run._extract_graphsignal_flags(['vllm', '--enable-otel']),
            (False, None, None, False, ['vllm', '--enable-otel']))

    def test_extracts_metrics_port_space_form(self):
        self.assertEqual(
            graphsignal_run._extract_graphsignal_flags(
                ['--metrics-port', '8000', 'trtllm-serve', 'm']),
            (False, 8000, None, False, ['trtllm-serve', 'm']))

    def test_extracts_metrics_port_equals_form(self):
        self.assertEqual(
            graphsignal_run._extract_graphsignal_flags(
                ['--metrics-port=8000', 'trtllm-serve', 'm']),
            (False, 8000, None, False, ['trtllm-serve', 'm']))

    def test_extracts_both_leading_flags(self):
        self.assertEqual(
            graphsignal_run._extract_graphsignal_flags(
                ['--enable-otel', '--metrics-port', '9001', 'vllm', 'serve']),
            (True, 9001, None, False, ['vllm', 'serve']))

    def test_extracts_cuda_graph_trace_space_form(self):
        self.assertEqual(
            graphsignal_run._extract_graphsignal_flags(
                ['--cuda-graph-trace', 'node', 'vllm', 'serve']),
            (False, None, 'node', False, ['vllm', 'serve']))

    def test_extracts_cuda_graph_trace_equals_form(self):
        self.assertEqual(
            graphsignal_run._extract_graphsignal_flags(
                ['--cuda-graph-trace=graph', 'python', 'app.py']),
            (False, None, 'graph', False, ['python', 'app.py']))

    def test_cuda_graph_trace_after_command_left_for_workload(self):
        self.assertEqual(
            graphsignal_run._extract_graphsignal_flags(
                ['vllm', 'serve', '--cuda-graph-trace', 'node']),
            (False, None, None, False, ['vllm', 'serve', '--cuda-graph-trace', 'node']))

    def test_extracts_leading_auto_flags(self):
        self.assertEqual(
            graphsignal_run._extract_graphsignal_flags(['--auto-flags', 'vllm', 'serve']),
            (False, None, None, True, ['vllm', 'serve']))

    def test_auto_flags_after_command_left_for_workload(self):
        self.assertEqual(
            graphsignal_run._extract_graphsignal_flags(['vllm', '--auto-flags']),
            (False, None, None, False, ['vllm', '--auto-flags']))

    def test_extracts_all_leading_flags(self):
        self.assertEqual(
            graphsignal_run._extract_graphsignal_flags(
                ['--enable-otel', '--metrics-port', '9001', '--cuda-graph-trace',
                 'node', '--auto-flags', 'vllm', 'serve']),
            (True, 9001, 'node', True, ['vllm', 'serve']))

    def test_invalid_cuda_graph_trace_exits(self):
        with self.assertRaises(SystemExit):
            graphsignal_run._extract_graphsignal_flags(
                ['--cuda-graph-trace', 'both', 'vllm', 'serve'])

    def test_cuda_graph_trace_missing_value_exits(self):
        with self.assertRaises(SystemExit):
            graphsignal_run._extract_graphsignal_flags(['--cuda-graph-trace'])

    def test_metrics_port_after_command_left_for_workload(self):
        self.assertEqual(
            graphsignal_run._extract_graphsignal_flags(
                ['vllm', 'serve', '--metrics-port', '8000']),
            (False, None, None, False, ['vllm', 'serve', '--metrics-port', '8000']))

    def test_invalid_metrics_port_exits(self):
        with self.assertRaises(SystemExit):
            graphsignal_run._extract_graphsignal_flags(
                ['--metrics-port', 'abc', 'vllm', 'serve'])

    def test_metrics_port_missing_value_exits(self):
        with self.assertRaises(SystemExit):
            graphsignal_run._extract_graphsignal_flags(['--metrics-port'])


class MainEnvTest(unittest.TestCase):
    def test_main_passes_cuda_graph_trace_to_launcher(self):
        created = []
        orig_init = VllmLauncher.__init__

        def capture_init(self, args, enable_otel=False, metrics_port=None,
                         cuda_graph_trace=None, auto_flags=False):
            created.append(cuda_graph_trace)
            return orig_init(self, args, enable_otel=enable_otel,
                             metrics_port=metrics_port,
                             cuda_graph_trace=cuda_graph_trace,
                             auto_flags=auto_flags)

        with patch.object(VllmLauncher, '__init__', capture_init), \
             patch.object(VllmLauncher, 'match', return_value=True), \
             patch.object(VllmLauncher, 'launch'), \
             patch.object(SglangLauncher, 'match', return_value=False), \
             patch.object(TrtllmLauncher, 'match', return_value=False), \
             patch.object(FallbackLauncher, 'match', return_value=False), \
             patch.object(sys, 'argv',
                          ['graphsignal-run', '--cuda-graph-trace', 'node', 'vllm', 'serve']):
            graphsignal_run.main()
            self.assertEqual(created, ['node'])

    def test_main_passes_none_cuda_graph_trace_when_flag_absent(self):
        created = []
        orig_init = VllmLauncher.__init__

        def capture_init(self, args, enable_otel=False, metrics_port=None,
                         cuda_graph_trace=None, auto_flags=False):
            created.append(cuda_graph_trace)
            return orig_init(self, args, enable_otel=enable_otel,
                             metrics_port=metrics_port,
                             cuda_graph_trace=cuda_graph_trace,
                             auto_flags=auto_flags)

        with patch.object(VllmLauncher, '__init__', capture_init), \
             patch.object(VllmLauncher, 'match', return_value=True), \
             patch.object(VllmLauncher, 'launch'), \
             patch.object(SglangLauncher, 'match', return_value=False), \
             patch.object(TrtllmLauncher, 'match', return_value=False), \
             patch.object(FallbackLauncher, 'match', return_value=False), \
             patch.object(sys, 'argv', ['graphsignal-run', 'vllm', 'serve']):
            graphsignal_run.main()
            self.assertEqual(created, [None])

    def test_main_passes_auto_flags_to_launcher(self):
        created = []
        orig_init = VllmLauncher.__init__

        def capture_init(self, args, enable_otel=False, metrics_port=None,
                         cuda_graph_trace=None, auto_flags=False):
            created.append(auto_flags)
            return orig_init(self, args, enable_otel=enable_otel,
                             metrics_port=metrics_port,
                             cuda_graph_trace=cuda_graph_trace,
                             auto_flags=auto_flags)

        with patch.object(VllmLauncher, '__init__', capture_init), \
             patch.object(VllmLauncher, 'match', return_value=True), \
             patch.object(VllmLauncher, 'launch'), \
             patch.object(SglangLauncher, 'match', return_value=False), \
             patch.object(TrtllmLauncher, 'match', return_value=False), \
             patch.object(FallbackLauncher, 'match', return_value=False), \
             patch.object(sys, 'argv',
                          ['graphsignal-run', '--auto-flags', 'vllm', 'serve']):
            graphsignal_run.main()
            self.assertEqual(created, [True])


if __name__ == '__main__':
    unittest.main()
