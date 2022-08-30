import logging
import time
import random
from threading import Lock

logger = logging.getLogger('graphsignal')

# global sampling lock
_sampling_lock = Lock()

class TraceSampler:
    MAX_ENSURED_SPANS = 2
    PREDEFINED_SPANS = [1, 10, 100, 1000]
    MIN_INTERVAL_SEC = 60

    def __init__(self):
        self._current_span = 0
        self._ensured_span_count = 0
        self._start_ts = time.time()
        self._last_sample_ts = None
        self._span_filter = {span for span in TraceSampler.PREDEFINED_SPANS}
        self._interval_mode = False
        self._profiled_count = 0

    def lock(self, ensure=False):
        self._current_span += 1

        if _sampling_lock.locked():
            return False

        now = time.time()

        if not self._interval_mode and now - self._start_ts > self.MIN_INTERVAL_SEC:
            # switch to interval-based sampling and sample first interval
            self._interval_mode = True
            _sampling_lock.acquire(blocking=False)
            return True

        if ensure:
            self._ensured_span_count += 1
            if self._ensured_span_count <= TraceSampler.MAX_ENSURED_SPANS:
                _sampling_lock.acquire(blocking=False)
                return True

        if not self._interval_mode:
            if self._current_span in self._span_filter:
                _sampling_lock.acquire(blocking=False)
                return True
        else:
            if not self._last_sample_ts:
                last_interval_sec = now - self._start_ts
            else:
                last_interval_sec = now - self._last_sample_ts
            if last_interval_sec > self.MIN_INTERVAL_SEC:
                self._ensured_span_count = 0
                _sampling_lock.acquire(blocking=False)
                return True

        return False

    def unlock(self):
        if not _sampling_lock.locked():
            return
        self._last_sample_ts = time.time()
        _sampling_lock.release()

    def should_profile(self):
        # only profile few initial spans
        if not self._interval_mode and self._current_span in self._span_filter:
            return True
        return False