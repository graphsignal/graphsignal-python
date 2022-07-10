import logging
import time
import random
from threading import Lock

logger = logging.getLogger('graphsignal')

# global profiling lock
_profiling_lock = Lock()

class ProfileScheduler:
    MAX_ENSURED_SPANS = 10
    PREDEFINED_SPANS = [1, 5, 10, 25, 100, 250, 1000]
    MIN_INTERVAL_SEC = 60
    MIN_INTERVAL_SPANS = 20

    def __init__(self):
        self._current_span = -1
        self._ensured_inference_count = 0
        self._last_profiled_ts = None
        self._last_profiled_span = None
        self._span_filter = {span for span in ProfileScheduler.PREDEFINED_SPANS}

    def lock(self, ensure=False):
        self._current_span += 1

        if _profiling_lock.locked():
            return False

        should_acquire = False

        if ensure:
            self._ensured_inference_count += 1
            if self._ensured_inference_count <= ProfileScheduler.MAX_ENSURED_SPANS:
                should_acquire = True

        if self._current_span + 1 in self._span_filter:
            should_acquire = True

        if self._last_profiled_ts:
            last_interval_sec = time.time() - self._last_profiled_ts
            last_interval_spans = self._current_span - self._last_profiled_span
            if last_interval_sec > self.MIN_INTERVAL_SEC and last_interval_spans > self.MIN_INTERVAL_SPANS:
                self._span_filter = {} # switch to interval-based profiling
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
        self._last_profiled_span = self._current_span
        _profiling_lock.release()
