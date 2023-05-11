import logging

logger = logging.getLogger('graphsignal')


class MissingValueSampler:
    MISSING_COUNTERS = {
        'null_count',
        'nan_count',
        'inf_count',
        'empty_count'
    }

    def __init__(self):
        pass

    def sample(self, data_name, counts):
        for counter_name in self.MISSING_COUNTERS:
            if counter_name in counts:
                if counts[counter_name] > 0:
                    return True

        if 'element_count' in counts:
            if counts['element_count'] == 0:
                return True

        return False