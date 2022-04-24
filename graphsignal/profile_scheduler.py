import logging
import time
import random
from threading import Lock

logger = logging.getLogger('graphsignal')

MAX_SCHEDULERS = 10

# global profiling lock
_profiling_lock = Lock()
_schedulers = {}


class ProfileScheduler(object):
    __slots__ = [
        '_total_step_count',
        '_ensured_step_count',
        '_step_filter',
        '_last_interval_ts'
    ]

    DEFAULT_SPANS = [10, 100, 1000]
    MAX_ENSURED_SPANS = 10
    MIN_SPAN_INTERVAL_SEC = 20

    def __init__(self):
        self._total_step_count = 0
        self._ensured_step_count = 0
        self._last_interval_ts = 0
        self._step_filter = {
            step: True for step in ProfileScheduler.DEFAULT_SPANS}

    def lock(self, ensure=False):
        self._total_step_count += 1

        if _profiling_lock.locked():
            return False

        if ensure:
            self._ensured_step_count += 1
            if self._ensured_step_count > ProfileScheduler.MAX_ENSURED_SPANS:
                return False
        else:
            # check if step index matches default step indexes
            if self._total_step_count not in self._step_filter:
                # skip first step
                if self._total_step_count == 1:
                    return False

                # comply with interval between steps
                if self._last_interval_ts > time.time() - \
                        self.MIN_SPAN_INTERVAL_SEC:
                    return False

        # set global lock
        return _profiling_lock.acquire(blocking=False)

    def unlock(self):
        if not _profiling_lock.locked():
            return
        self._last_interval_ts = time.time()
        _profiling_lock.release()


def select_scheduler(run_phase):
    if run_phase is None:
        run_phase = 0

    if run_phase in _schedulers:
        return _schedulers[run_phase]
    else:
        if len(_schedulers) < MAX_SCHEDULERS:
            scheduler = _schedulers[run_phase] = ProfileScheduler()
            return scheduler
        else:
            return random.choice(list(_schedulers.values()))
