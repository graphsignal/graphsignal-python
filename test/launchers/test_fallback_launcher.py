import os
import sys
import unittest
from unittest.mock import MagicMock, patch

from graphsignal.launchers import fallback_launcher as fallback_mod
from graphsignal.launchers.fallback_launcher import FallbackLauncher


class FallbackMatchTest(unittest.TestCase):
    def test_always_matches(self):
        self.assertTrue(FallbackLauncher([]).match())
        self.assertTrue(FallbackLauncher(['python', 'app.py']).match())
        self.assertTrue(FallbackLauncher(['anything', '--whatever']).match())


class FallbackLaunchTest(unittest.TestCase):
    """FallbackLauncher always exec's into a fresh process:
       * resolvable executable → `os.execv(resolved, [resolved, ...])`
       * otherwise → `os.execv(sys.executable, [py, '-m', name, ...])`
    """

    def setUp(self):
        self.setup_env = patch.object(fallback_mod.CuptiProfiler, 'setup_env_vars', return_value=True)
        self.start_watcher = patch.object(fallback_mod, 'start_watcher', return_value=MagicMock())
        self.setup_env_m = self.setup_env.start()
        self.start_watcher_m = self.start_watcher.start()

    def tearDown(self):
        self.start_watcher.stop()
        self.setup_env.stop()

    def test_no_args_exits(self):
        with self.assertRaises(SystemExit) as cm:
            FallbackLauncher([]).launch()
        self.assertEqual(cm.exception.code, 1)
        self.start_watcher_m.assert_not_called()

    def test_absolute_py_path_execs_directly(self):
        # Absolute path to an existing file resolves and execs directly;
        # whether it actually runs depends on the file's executable bit
        # and shebang (the launcher no longer wraps it in `python ...`).
        with patch.object(fallback_mod, '_resolve', return_value='/abs/my_script.py'), \
             patch('os.execv') as execv_m:
            FallbackLauncher(['/abs/my_script.py', '--flag']).launch()
        # Generic workload without --port → no Prometheus scrape port.
        self.start_watcher_m.assert_called_once_with(os.getpid(), metrics_port=None)
        execv_m.assert_called_once_with(
            '/abs/my_script.py', ['/abs/my_script.py', '--flag'])

    def test_execv_permission_error_suggests_python_prefix(self):
        # Non-executable script (or missing shebang) → execv raises
        # PermissionError; launcher prints a friendly hint and exits 1.
        with patch.object(fallback_mod, '_resolve', return_value='/abs/my_script.py'), \
             patch('os.execv', side_effect=PermissionError), \
             patch('builtins.print') as print_m:
            with self.assertRaises(SystemExit) as cm:
                FallbackLauncher(['/abs/my_script.py']).launch()
        self.assertEqual(cm.exception.code, 1)
        self.assertEqual(print_m.call_count, 2)

    def test_executable_on_path_uses_execv(self):
        with patch.object(fallback_mod, '_resolve', return_value='/usr/bin/myapp'), \
             patch('os.execv') as execv_m:
            FallbackLauncher(['myapp', '--flag']).launch()
        self.start_watcher_m.assert_called_once_with(os.getpid(), metrics_port=None)
        execv_m.assert_called_once_with('/usr/bin/myapp', ['/usr/bin/myapp', '--flag'])

    def test_explicit_metrics_port_forwarded(self):
        with patch.object(fallback_mod, '_resolve', return_value='/usr/bin/myapp'), \
             patch('os.execv'):
            FallbackLauncher(['myapp'], metrics_port=9999).launch()
        self.start_watcher_m.assert_called_once_with(os.getpid(), metrics_port=9999)

    def test_metrics_port_from_workload_port_flag(self):
        with patch.object(fallback_mod, '_resolve', return_value='/usr/bin/myapp'), \
             patch('os.execv'):
            FallbackLauncher(['myapp', '--port', '8080']).launch()
        self.start_watcher_m.assert_called_once_with(os.getpid(), metrics_port=8080)

    def test_unresolved_runs_python_dash_m(self):
        with patch.object(fallback_mod, '_resolve', return_value=None), \
             patch('os.execv') as execv_m:
            FallbackLauncher(['my.module', '--flag']).launch()
        execv_m.assert_called_once_with(
            sys.executable,
            [sys.executable, '-m', 'my.module', '--flag'])


if __name__ == '__main__':
    unittest.main()
