import logging

import graphsignal
from graphsignal.proto import signals_pb2
from graphsignal.data.data_profiler import DataProfiler

logger = logging.getLogger('graphsignal')


class BuiltInTypesProfiler(DataProfiler):
    def __init__(self):
        super().__init__()

    def is_instance(self, data):
        return isinstance(data, (list, dict, tuple, set, str, int, float, complex)) or data is None

    def get_size(self, data):
        if isinstance(data, bytes):
            return (len(data), 'bytes')
        elif isinstance(data, str):
            return (len(data), 'char')
        else:
            return (self._get_list_size(data), 'elem')

    def _get_list_size(self, data):
        if isinstance(data, dict):
            return sum([self._get_list_size(elem) for elem in data.values()])
        elif isinstance(data, (list, tuple, set)):
            return sum([self._get_list_size(elem) for elem in data])
        else:
            return 1

    def compute_stats(self, data):
        data_stats = signals_pb2.DataStats()
        data_stats.data_type = type(data).__name__
        data_stats.size, _ = self.get_size(data)
        data_stats.num_null = _count_elems(data, lambda elem: elem is None)
        data_stats.num_nan = _count_elems(data, lambda elem: elem != elem)
        data_stats.num_inf = _count_elems(data, lambda elem: elem == float("inf"))
        data_stats.num_zero = _count_elems(data, lambda elem: elem == 0)
        data_stats.num_unique = _count_unique(data, set())
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


def _count_unique(data, unique_set):
    if isinstance(data, dict):
        for elem in data.values():
            _count_unique(elem, unique_set)
    elif isinstance(data, (list, tuple, set)):
        for elem in data:
            _count_unique(elem, unique_set)
    elif isinstance(data, (int, float, complex, str, bytes)):
        unique_set.add(data)
    elif data is None:
        unique_set.add(None)

    return len(unique_set)
