import glob
import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from graphsignal.launchers import supervisor
from graphsignal.launchers.supervisor import _SegmentWriter, launch_supervised


class SupervisorLaunchTest(unittest.TestCase):
    """launch_supervised spawns the workload as a child, tees its console to
    JSONL segments, targets the watcher at the child's pid, and exits with the
    child's exact status."""

    def setUp(self):
        self.shm_base = tempfile.mkdtemp()
        self.shm_patch = patch.object(supervisor, '_SHM_BASE', self.shm_base)
        self.watcher_patch = patch.object(
            supervisor, 'start_watcher', return_value=MagicMock())
        self.shm_patch.start()
        self.start_watcher_m = self.watcher_patch.start()

    def tearDown(self):
        self.watcher_patch.stop()
        self.shm_patch.stop()
        shutil.rmtree(self.shm_base, ignore_errors=True)

    def _log_records(self):
        records = []
        for path in sorted(glob.glob(
                os.path.join(self.shm_base, 'graphsignal_log_*', 'log_*.jsonl'))):
            with open(path) as f:
                records.extend(json.loads(line) for line in f if line.strip())
        return records

    def test_no_args_exits(self):
        with self.assertRaises(SystemExit) as cm:
            launch_supervised([])
        self.assertEqual(cm.exception.code, 1)
        self.start_watcher_m.assert_not_called()

    def test_mirrors_child_exit_code(self):
        with self.assertRaises(SystemExit) as cm:
            launch_supervised(['/bin/sh', '-c', 'exit 7'])
        self.assertEqual(cm.exception.code, 7)

    def test_mirrors_child_signal_death(self):
        # Child kills itself with SIGTERM → rc = -15. The supervisor re-raises
        # the same signal on itself; os.kill is mocked so the test process
        # survives and the 128+N fallback exit code is observed instead.
        with patch('os.kill') as kill_m:
            with self.assertRaises(SystemExit) as cm:
                launch_supervised(['/bin/sh', '-c', 'kill -TERM $$'])
        self.assertEqual(cm.exception.code, 128 + 15)
        kill_m.assert_called_once_with(os.getpid(), 15)

    def test_tees_stdout_and_stderr_to_jsonl(self):
        with self.assertRaises(SystemExit) as cm:
            launch_supervised(['/bin/sh', '-c', 'echo out_line; echo err_line 1>&2'])
        self.assertEqual(cm.exception.code, 0)

        records = self._log_records()
        by_stream = {r['stream']: r for r in records if 'line' in r}
        self.assertEqual(by_stream['stdout']['line'], 'out_line')
        self.assertEqual(by_stream['stderr']['line'], 'err_line')
        for record in records:
            self.assertGreater(record['ts'], 0)

    def test_writes_exit_code_record_on_nonzero_exit(self):
        with self.assertRaises(SystemExit):
            launch_supervised(['/bin/sh', '-c', 'echo bye; exit 5'])
        records = self._log_records()
        exit_records = [r for r in records if 'exit_code' in r]
        self.assertEqual(len(exit_records), 1)
        self.assertEqual(exit_records[0]['exit_code'], 5)

    def test_writes_signal_record_on_signal_death(self):
        with patch('os.kill'):
            with self.assertRaises(SystemExit):
                launch_supervised(['/bin/sh', '-c', 'kill -TERM $$'])
        records = self._log_records()
        signal_records = [r for r in records if 'signal' in r]
        self.assertEqual(len(signal_records), 1)
        self.assertEqual(signal_records[0]['signal'], 15)

    def test_no_status_record_on_clean_exit(self):
        with self.assertRaises(SystemExit):
            launch_supervised(['/bin/sh', '-c', 'echo ok; exit 0'])
        records = self._log_records()
        self.assertFalse([r for r in records if 'exit_code' in r or 'signal' in r])

    def test_watcher_targets_child_pid_with_passthrough_kwargs(self):
        with self.assertRaises(SystemExit):
            launch_supervised(['/bin/sh', '-c', 'exit 0'],
                              workload_id='abc123', otel_collector_port=4317,
                              metrics_port=8000, metrics_path='/m', metrics_host='h')

        self.start_watcher_m.assert_called_once()
        (pid,), kwargs = self.start_watcher_m.call_args
        self.assertNotEqual(pid, os.getpid())
        self.assertEqual(kwargs, {
            'workload_id': 'abc123', 'otel_collector_port': 4317,
            'metrics_port': 8000, 'metrics_path': '/m', 'metrics_host': 'h'})
        # The log dir is derived from the same child pid the watcher targets.
        self.assertTrue(os.path.isdir(
            os.path.join(self.shm_base, f'graphsignal_log_{pid}')))

    def test_capture_disabled_when_shm_base_missing(self):
        missing = os.path.join(self.shm_base, 'nonexistent')
        with patch.object(supervisor, '_SHM_BASE', missing):
            with self.assertRaises(SystemExit) as cm:
                launch_supervised(['/bin/sh', '-c', 'echo hi; exit 0'])
        self.assertEqual(cm.exception.code, 0)
        self.assertFalse(os.path.isdir(missing))

    def test_permission_error_suggests_python_prefix(self):
        script = os.path.join(self.shm_base, 'noexec.py')
        with open(script, 'w') as f:
            f.write('print("hi")\n')
        with patch('builtins.print') as print_m:
            with self.assertRaises(SystemExit) as cm:
                launch_supervised([script])
        self.assertEqual(cm.exception.code, 1)
        self.assertEqual(print_m.call_count, 2)
        self.start_watcher_m.assert_not_called()


class SegmentWriterTest(unittest.TestCase):
    def setUp(self):
        self.log_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.log_dir, ignore_errors=True)

    def _segments(self):
        return sorted(os.listdir(self.log_dir))

    def test_writes_jsonl_records(self):
        writer = _SegmentWriter(self.log_dir)
        writer.write('stdout', 'hello', ts_ns=123)
        writer.close()
        with open(os.path.join(self.log_dir, 'log_0.jsonl')) as f:
            record = json.loads(f.read())
        self.assertEqual(record, {'ts': 123, 'stream': 'stdout', 'line': 'hello'})

    def test_rotation_keeps_max_segments(self):
        writer = _SegmentWriter(self.log_dir, segment_size_limit=100, max_segments=2)
        for i in range(20):
            writer.write('stdout', 'line-%02d-%s' % (i, 'x' * 40))
        writer.close()
        segments = self._segments()
        self.assertEqual(len(segments), 2)
        # The newest lines survive; the oldest segments were pruned.
        self.assertNotIn('log_0.jsonl', segments)

    def test_failed_writes_drop_and_record_marker(self):
        missing_dir = os.path.join(self.log_dir, 'gone')
        writer = _SegmentWriter(missing_dir)
        writer.write('stdout', 'dropped-1')
        writer.write('stdout', 'dropped-2')

        os.makedirs(missing_dir)
        writer.write('stdout', 'kept')
        writer.close()

        with open(os.path.join(missing_dir, 'log_0.jsonl')) as f:
            records = [json.loads(line) for line in f]
        self.assertEqual(records[0]['dropped'], 2)
        self.assertEqual(records[1]['line'], 'kept')

    def test_write_after_close_is_ignored(self):
        writer = _SegmentWriter(self.log_dir)
        writer.write('stdout', 'before')
        writer.close()
        writer.write('stdout', 'after')
        self.assertEqual(self._segments(), ['log_0.jsonl'])
        with open(os.path.join(self.log_dir, 'log_0.jsonl')) as f:
            lines = f.read().splitlines()
        self.assertEqual(len(lines), 1)


if __name__ == '__main__':
    unittest.main()
