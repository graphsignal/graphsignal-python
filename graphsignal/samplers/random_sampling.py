import logging
import time

logger = logging.getLogger('graphsignal')

class RandomSampler:
    MAX_EVENTS_PER_MIN = 1000

    def __init__(self):
        self._num_events = 0
        self._last_reset_ts = time.time()

    def sample(self):
        now = time.time()
        if now - self._last_reset_ts > 60:
            self._num_events = 0
            self._last_reset_ts = now

        if self._num_events >= self.MAX_EVENTS_PER_MIN:
            return False

        self._num_events += 1
        return True