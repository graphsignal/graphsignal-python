import logging
import time
import random
from threading import Lock

logger = logging.getLogger('graphsignal')

# global profiling lock
_profiling_lock = Lock()

class ProfileScheduler:
    MAX_ENSURED_SPANS = 10
    PREDEFINED_SPANS = [1, 10, 100, 1000]
    MIN_INTERVAL_SEC = 60

    def __init__(self):
        self._current_span = 0
        self._ensured_span_count = 0
        self._start_ts = time.time()
        self._last_profile_ts = None
        self._last_profiled_span = None
        self._span_filter = {span for span in ProfileScheduler.PREDEFINED_SPANS}
        self._interval_mode = False

    def lock(self, ensure=False):
        self._current_span += 1

        if _profiling_lock.locked():
            return False

        now = time.time()

        if not self._interval_mode and now - self._start_ts > self.MIN_INTERVAL_SEC:
            # switch to interval-based profiling and profile first interval
            self._interval_mode = True
            _profiling_lock.acquire(blocking=False)
            return True

        if ensure:
            self._ensured_span_count += 1
            if self._ensured_span_count <= ProfileScheduler.MAX_ENSURED_SPANS:
                _profiling_lock.acquire(blocking=False)
                return True

        if not self._interval_mode:
            if self._current_span in self._span_filter:
                _profiling_lock.acquire(blocking=False)
                return True
        else:
            if not self._last_profile_ts:
                last_interval_sec = now - self._start_ts
            else:
                last_interval_sec = now - self._last_profile_ts
            logger.debug(last_interval_sec)
            if last_interval_sec > self.MIN_INTERVAL_SEC:
                _profiling_lock.acquire(blocking=False)
                return True

        return False

    def unlock(self):
        if not _profiling_lock.locked():
            return
        self._last_profile_ts = time.time()
        _profiling_lock.release()
