import os
import time
import unittest

from graphsignal.sdk.pid_watcher import PidWatcher


class _RecordingListener:
    def __init__(self):
        self.known = []
        self.created = 0
        self.terminated = 0

    def on_target_known(self, pid):
        self.known.append(pid)

    def on_target_created(self, args):
        self.created += 1

    def on_target_terminated(self):
        self.terminated += 1


class PidWatcherTest(unittest.TestCase):
    def test_on_target_known_emitted_at_setup_before_polling(self):
        # on_target_known fires synchronously from setup(), so a consumer of a
        # pid-derived artifact is wired up even if the target never polls alive.
        listener = _RecordingListener()
        watcher = PidWatcher(target_pid=os.getpid(), poll_interval=0.05)
        watcher.add_listener(listener)
        try:
            watcher.setup()
            self.assertEqual(listener.known, [os.getpid()])
        finally:
            watcher.shutdown()

    def test_on_target_known_fires_even_for_dead_target(self):
        # Pick a pid that is (almost certainly) not running.
        dead_pid = 999999
        listener = _RecordingListener()
        watcher = PidWatcher(target_pid=dead_pid, poll_interval=0.05)
        watcher.add_listener(listener)
        try:
            watcher.setup()
            # Known fires from setup regardless of liveness; the poll then sees
            # it dead and reports terminated without ever reporting created.
            self.assertEqual(listener.known, [dead_pid])
            time.sleep(0.2)
            self.assertEqual(listener.created, 0)
            self.assertGreaterEqual(listener.terminated, 1)
        finally:
            watcher.shutdown()

    def test_listener_exception_in_on_target_known_does_not_break_setup(self):
        class _Boom:
            def on_target_known(self, pid):
                raise RuntimeError('boom')

        watcher = PidWatcher(target_pid=os.getpid(), poll_interval=0.05)
        watcher.add_listener(_Boom())
        try:
            watcher.setup()  # must not raise
        finally:
            watcher.shutdown()


if __name__ == '__main__':
    unittest.main()
