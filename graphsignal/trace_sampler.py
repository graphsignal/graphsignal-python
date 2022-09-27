import logging
import time
import random
from threading import Lock

logger = logging.getLogger('graphsignal')

# global sampling lock
_sampling_lock = Lock()

class TraceSampler:
    MIN_INTERVAL_SEC = 60

    def __init__(self):
        self._current_trace_idx = 0
        self._trace_counters = {}
        self._last_reset_ts = time.time()

    def current_trace_idx(self):
        return self._current_trace_idx

    def lock(self, group, limit_per_interval=1, include_trace_idx=None):
        self._current_trace_idx += 1

        if _sampling_lock.locked():
            return False

        now = time.time()

        # reset counters
        if now - self._last_reset_ts > self.MIN_INTERVAL_SEC:
            self._trace_counters = {}
            self._last_reset_ts = time.time()

        # check counters
        if group in self._trace_counters:
            self._trace_counters[group] += 1
        else:
            self._trace_counters[group] = 1
        if self._trace_counters[group] <= limit_per_interval:
            _sampling_lock.acquire(blocking=False)
            return True

        # check include_trace_idx
        if include_trace_idx is not None and self._current_trace_idx in include_trace_idx:
            _sampling_lock.acquire(blocking=False)
            return True

        return False

    def unlock(self):
        if not _sampling_lock.locked():
            return
        _sampling_lock.release()
