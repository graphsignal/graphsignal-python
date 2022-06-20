import logging
import time
import random
from threading import Lock

logger = logging.getLogger('graphsignal')

MAX_SCHEDULERS = 10

# global profiling lock
_profiling_lock = Lock()
_schedulers = {}


class ProfileScheduler:
    MAX_ENSURED_STEPS = 10
    PREDEFINED_STEPS = [2, 10, 25, 100, 250, 1000]
    MIN_INTERVAL_SEC = 60
    MIN_INTERVAL_STEPS = 20

    def __init__(self):
        self._current_step = -1
        self._ensured_step_count = 0
        self._last_profiled_ts = None
        self._last_profiled_step = None

        # Avoid profiling at the same time with workers
        step_shift = random.randint(0, 5)
        self._step_filter = {
            step + step_shift for step in ProfileScheduler.PREDEFINED_STEPS}

    def lock(self, ensure=False):
        self._current_step += 1

        if _profiling_lock.locked():
            return False

        should_acquire = False

        if ensure:
            self._ensured_step_count += 1
            if self._ensured_step_count <= ProfileScheduler.MAX_ENSURED_STEPS:
                should_acquire = True

        if self._current_step + 1 in self._step_filter:
            should_acquire = True

        if self._last_profiled_ts:
            last_interval_sec = time.time() - self._last_profiled_ts
            last_interval_steps = self._current_step - self._last_profiled_step
            if last_interval_sec > self.MIN_INTERVAL_SEC and last_interval_steps > self.MIN_INTERVAL_STEPS:
                self._step_filter = {} # switch to interval-based profiling
                should_acquire = True

        if should_acquire:
            _profiling_lock.acquire(blocking=False)
            return True
        else:
            return False

    def unlock(self):
        if not _profiling_lock.locked():
            return
        self._last_profiled_ts = time.time()
        self._last_profiled_step = self._current_step
        _profiling_lock.release()


def select_scheduler(scope):
    if scope is None:
        scope = 0

    if scope in _schedulers:
        return _schedulers[scope]
    else:
        if len(_schedulers) < MAX_SCHEDULERS:
            scheduler = _schedulers[scope] = ProfileScheduler()
            return scheduler
        else:
            return random.choice(list(_schedulers.values()))
