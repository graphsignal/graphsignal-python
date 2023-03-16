import logging
import contextvars
import time


logger = logging.getLogger('graphsignal')


class LatencyOutlierDetector:
    MIN_VALUES = 100
    MAX_VALUES = 1000

    def __init__(self):
        self.reset()

    def reset(self):
        self._sum = 0
        self._sum2 = 0
        self._count = 0

    def update(self, value):
        self._sum += value
        self._sum2 += value * value
        self._count += 1

    def detect(self, value):
        if self._count < self.MIN_VALUES:
            return False
        elif self._count > self.MAX_VALUES:
            self.reset()
            return False
        mean = self._sum / self._count
        std = (self._sum2 / self._count - mean * mean) ** 0.5
        return value > mean + 6 * std
