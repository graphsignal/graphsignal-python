import sys
import unittest
from unittest.mock import patch

from graphsignal.launchers import fallback_launcher as fallback_mod
from graphsignal.launchers.command_utils import hash_workload_id
from graphsignal.launchers.fallback_launcher import FallbackLauncher


class FallbackMatchTest(unittest.TestCase):
    def test_always_matches(self):
        self.assertTrue(FallbackLauncher([]).match())
        self.assertTrue(FallbackLauncher(['python', 'app.py']).match())
        self.assertTrue(FallbackLauncher(['anything', '--whatever']).match())


class FallbackLaunchTest(unittest.TestCase):
    """FallbackLauncher hands the workload to the supervisor:
       * resolvable executable → `launch_supervised([resolved, ...])`
       * otherwise → `launch_supervised([py, '-m', name, ...])`
    """

    def setUp(self):
        self.setup_env = patch.object(
            fallback_mod.CuptiProfiler, 'setup_env_vars', return_value=True)
        self.launch = patch.object(fallback_mod, 'launch_supervised')
        self.setup_env_m = self.setup_env.start()
        self.launch_m = self.launch.start()

    def tearDown(self):
        self.launch.stop()
        self.setup_env.stop()

    def test_no_args_exits(self):
        with self.assertRaises(SystemExit) as cm:
            FallbackLauncher([]).launch()
        self.assertEqual(cm.exception.code, 1)
        self.launch_m.assert_not_called()

    def test_absolute_py_path_launches_directly(self):
        # Absolute path to an existing file resolves and launches directly;
        # whether it actually runs depends on the file's executable bit
        # and shebang (the launcher no longer wraps it in `python ...`).
        with patch.object(fallback_mod, '_resolve', return_value='/abs/my_script.py'):
            FallbackLauncher(['/abs/my_script.py', '--flag']).launch()
        # Generic workload without --port → no Prometheus scrape port.
        self.launch_m.assert_called_once_with(
            ['/abs/my_script.py', '--flag'],
            workload_id=hash_workload_id(['/abs/my_script.py', '--flag']),
            metrics_port=None)

    def test_executable_on_path_launches(self):
        with patch.object(fallback_mod, '_resolve', return_value='/usr/bin/myapp'):
            FallbackLauncher(['myapp', '--flag']).launch()
        self.launch_m.assert_called_once_with(
            ['/usr/bin/myapp', '--flag'],
            workload_id=hash_workload_id(['myapp', '--flag']),
            metrics_port=None)

    def test_explicit_metrics_port_forwarded(self):
        with patch.object(fallback_mod, '_resolve', return_value='/usr/bin/myapp'):
            FallbackLauncher(['myapp'], metrics_port=9999).launch()
        self.launch_m.assert_called_once_with(
            ['/usr/bin/myapp'],
            workload_id=hash_workload_id(['myapp']),
            metrics_port=9999)

    def test_metrics_port_from_workload_port_flag(self):
        with patch.object(fallback_mod, '_resolve', return_value='/usr/bin/myapp'):
            FallbackLauncher(['myapp', '--port', '8080']).launch()
        self.launch_m.assert_called_once_with(
            ['/usr/bin/myapp', '--port', '8080'],
            workload_id=hash_workload_id(['myapp', '--port', '8080']),
            metrics_port=8080)

    def test_unresolved_runs_python_dash_m(self):
        with patch.object(fallback_mod, '_resolve', return_value=None):
            FallbackLauncher(['my.module', '--flag']).launch()
        self.launch_m.assert_called_once_with(
            [sys.executable, '-m', 'my.module', '--flag'],
            workload_id=hash_workload_id(['my.module', '--flag']),
            metrics_port=None)


if __name__ == '__main__':
    unittest.main()
