import unittest
import logging
import sys
import os
import subprocess
import tempfile
from unittest.mock import patch

import graphsignal
import graphsignal.sdk

logger = logging.getLogger('graphsignal')


class SdkConfigureTest(unittest.TestCase):
    def setUp(self):
        if len(logger.handlers) == 0:
            logger.addHandler(logging.StreamHandler(sys.stdout))
        graphsignal.sdk.configure(
            api_key='k1',
            debug_mode=True)
        graphsignal.sdk.sdk()._auto_tick = False

    def tearDown(self):
        graphsignal.sdk.shutdown()

    def test_configure(self):
        self.assertEqual(graphsignal.sdk.sdk().api_key(), 'k1')
        self.assertEqual(graphsignal.sdk.sdk().debug_mode, True)
        self.assertEqual(graphsignal.sdk.sdk().target_pid(), os.getpid())

    def test_sdk_accessor_returns_singleton(self):
        from graphsignal.sdk.sdk import Sdk
        self.assertIsInstance(graphsignal.sdk.sdk(), Sdk)
        self.assertIs(graphsignal.sdk.sdk(), graphsignal.sdk.sdk())
        # Same instance survives across calls.
        first = graphsignal.sdk.sdk()
        self.assertIs(graphsignal.sdk.sdk(), first)

    def test_sdk_raises_when_not_configured(self):
        graphsignal.sdk.shutdown()
        try:
            with self.assertRaises(RuntimeError):
                graphsignal.sdk.sdk()
        finally:
            # Re-configure so tearDown's shutdown is a no-op-friendly state.
            graphsignal.sdk.configure(api_key='k1', debug_mode=True)
            graphsignal.sdk.sdk()._auto_tick = False

    def test_fork_child_shuts_down_sdk(self):
        if not hasattr(os, 'fork'):
            return

        check_script = """
import sys
import os
import time

try:
    import graphsignal
    import graphsignal.sdk
    graphsignal.sdk.configure(api_key='test-fork-key', debug_mode=True)
    graphsignal.sdk.sdk()._auto_tick = False

    pid = os.fork()

    if pid == 0:
        time.sleep(0.2)
        # After fork(), the SDK in the child is expected to have shut itself
        # down (via the os.register_at_fork(after_in_child=...) hook).
        # `is_configured()` is False after shutdown completes.
        if not graphsignal.sdk.is_configured():
            sys.exit(0)
        sys.exit(1)
    else:
        _, status = os.waitpid(pid, 0)
        sys.exit(os.WEXITSTATUS(status) if os.WIFEXITED(status) else 1)
except Exception as e:
    import traceback
    print(f"ERROR: {e}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    sys.exit(99)
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(check_script)
            script_path = f.name

        try:
            env = os.environ.copy()
            env.pop('CUDA_INJECTION64_PATH', None)
            env.pop('GRAPHSIGNAL_API_KEY', None)
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            env['PYTHONPATH'] = project_root + os.pathsep + env.get('PYTHONPATH', '')
            result = subprocess.run(
                [sys.executable, script_path],
                env=env,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                self.fail(
                    "Fork-shutdown test failed with exit code "
                    f"{result.returncode}: stdout={result.stdout!r}, stderr={result.stderr!r}"
                )
        finally:
            os.unlink(script_path)


class WatchTest(unittest.TestCase):
    def test_watch_sets_cupti_env_and_spawns_watcher(self):
        sentinel_popen = object()
        # _start_watcher and _CuptiProfiler are aliases bound into graphsignal/__init__.py
        # at import time, so patch them where the alias lives.
        with patch('graphsignal._start_watcher',
                   return_value=sentinel_popen) as mock_start, \
             patch.object(graphsignal._CuptiProfiler, 'setup_env_vars',
                          return_value=True) as mock_setup_env:
            result = graphsignal.watch(otel_collector_port=4317)

        mock_setup_env.assert_called_once_with(cuda_graph_trace=None)
        mock_start.assert_called_once_with(
            os.getpid(), otel_collector_port=4317, metrics_port=None)
        self.assertIs(result, sentinel_popen)

    def test_watch_without_otel_port(self):
        with patch('graphsignal._start_watcher',
                   return_value=None) as mock_start, \
             patch.object(graphsignal._CuptiProfiler, 'setup_env_vars',
                          return_value=True):
            result = graphsignal.watch()

        mock_start.assert_called_once_with(
            os.getpid(), otel_collector_port=None, metrics_port=None)
        self.assertIsNone(result)

    def test_watch_forwards_metrics_port(self):
        with patch('graphsignal._start_watcher',
                   return_value=None) as mock_start, \
             patch.object(graphsignal._CuptiProfiler, 'setup_env_vars',
                          return_value=True):
            graphsignal.watch(metrics_port=8000)

        mock_start.assert_called_once_with(
            os.getpid(), otel_collector_port=None, metrics_port=8000)

    def test_watch_passes_cuda_graph_trace_to_setup_env_vars(self):
        with patch('graphsignal._start_watcher', return_value=None), \
             patch.object(graphsignal._CuptiProfiler, 'setup_env_vars',
                          return_value=True) as mock_setup_env:
            graphsignal.watch(cuda_graph_trace='node')
        mock_setup_env.assert_called_once_with(cuda_graph_trace='node')

    def test_watch_rejects_invalid_cuda_graph_trace(self):
        with patch('graphsignal._start_watcher', return_value=None):
            with self.assertRaises(ValueError):
                graphsignal.watch(cuda_graph_trace='both')
