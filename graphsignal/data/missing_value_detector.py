import logging

logger = logging.getLogger('graphsignal')


MISSING_COUNTERS = {
    'null_count',
    'nan_count',
    'inf_count',
    'empty_count'
}

class MissingValueDetector:
    def __init__(self):
        pass

    def detect(self, data_name, counts):
        for counter_name in MISSING_COUNTERS:
            if counter_name in counts:
                if counts[counter_name] > 0:
                    return True

        return False
