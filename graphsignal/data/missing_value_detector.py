import logging

logger = logging.getLogger('graphsignal')


MISSING_COUNTERS = {
    'null_count',
    'nan_count',
    'inf_count',
    'empty_count'
}

class MissingValueDetector:
    MIN_BASELINE_SIZE = 30

    def __init__(self):
        self._baselines = {}

    def detect(self, data_name, counts):
        has_missing = False
        for counter_name in MISSING_COUNTERS:
            if counter_name in counts:
                if counts[counter_name] > 0:
                    has_missing = True
                    break

        if data_name in self._baselines:
            baseline = self._baselines[data_name]
            baseline['size'] += 1
            if not baseline['has_missing'] and has_missing:
                baseline['has_missing'] = True
                if baseline['size'] > MissingValueDetector.MIN_BASELINE_SIZE:
                    return True
        else:
            self._baselines[data_name] = dict(size=1, has_missing=has_missing)

        return False
