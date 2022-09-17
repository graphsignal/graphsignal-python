import logging

import graphsignal
from graphsignal.proto import signals_pb2
from graphsignal.data.data_profiler import DataProfiler, add_counts

logger = logging.getLogger('graphsignal')


class BuiltInTypesProfiler(DataProfiler):
    def __init__(self):
        super().__init__()

    def is_instance(self, data):
        return isinstance(data, (list, dict, tuple, set, str, int, float, complex)) or data is None

    def compute_counts(self, data):
        counts = {}
        if isinstance(data, str):
            counts['char_count'] = len(data)
        elif isinstance(data, bytes):
            counts['byte_count'] = len(data)
        else:
            counts['element_count'] = _count_elems(data, lambda elem: True)
        counts['null_count'] = _count_elems(data, lambda elem: elem is None)
        counts['nan_count'] = _count_elems(data, lambda elem: elem != elem)
        counts['inf_count'] = _count_elems(data, lambda elem: elem == float("inf"))
        counts['zero_count'] = _count_elems(data, lambda elem: elem == 0)
        return counts

    def build_stats(self, data):
        counts = self.compute_counts(data)
        data_stats = signals_pb2.DataStats()
        data_stats.data_type = type(data).__name__
        add_counts(data_stats, counts)
        return data_stats


def _count_elems(data, filter_func):
    if isinstance(data, dict):
        return sum([_count_elems(elem, filter_func) for elem in data.values()])
    elif isinstance(data, (list, tuple, set)):
        return sum([_count_elems(elem, filter_func) for elem in data])
    elif isinstance(data, (int, float, complex, str, bytes)):
        return sum([filter_func(data)])
    elif data is None:
        return sum([filter_func(data)])
    return 0
