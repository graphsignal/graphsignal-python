import logging
import time

logger = logging.getLogger('graphsignal')

class TraceSampler:
    MAX_SAMPLES_PER_SECOND = 0.1
    EXTRA_SAMPLES = 100

    def __init__(self):
        self._num_sampled = {}
        self._start_ts = time.monotonic()

    def sample(self, group):
        seconds_since_start = time.monotonic() - self._start_ts

        num_sampled = self._num_sampled.get(group, 0)
        if num_sampled - self.EXTRA_SAMPLES < seconds_since_start * self.MAX_SAMPLES_PER_SECOND:
            self._num_sampled[group] = num_sampled + 1
            return True

        return False
