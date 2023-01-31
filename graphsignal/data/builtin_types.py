import logging
import sys

import graphsignal
from graphsignal.proto import signals_pb2
from graphsignal.data.base_data_profiler import BaseDataProfiler, DataStats

logger = logging.getLogger('graphsignal')


class BuiltInTypesProfiler(BaseDataProfiler):
    def __init__(self):
        super().__init__()

    def is_instance(self, data):
        return isinstance(data, (list, dict, tuple, set, str, int, float, complex)) or data is None

    def compute_stats(self, data):
        counts = {}
        counts['byte_count'] = _size(data)
        counts['element_count'] = _count_true(data, lambda elem: True)
        counts['null_count'] = _count_true(data, lambda elem: elem is None)
        counts['nan_count'] = _count_true(data, lambda elem: elem != elem)
        counts['inf_count'] = _count_true(data, lambda elem: elem == float("inf") or elem == float("-inf"))
        counts['zero_count'] = _count_true(data, lambda elem: elem == 0)
        counts['empty_count'] = _count_true(data, lambda elem: elem == '')
        counts['negative_count'] = _count_true(data, lambda elem: isinstance(elem, (int, float)) and elem < 0)
        counts['positive_count'] = _count_true(data, lambda elem: isinstance(elem, (int, float)) and elem > 0)
        return DataStats(type_name=type(data).__name__, shape=_shape(data), counts=counts)


def _shape(data):
    if isinstance(data, list):
        if len(data) == 0:
            return [0]
        else:
            return [len(data)] + _shape(data[0])
    return []


def _size(data):
    if isinstance(data, dict):
        return sum([_size(elem) for elem in data.values()])
    elif isinstance(data, (list, tuple, set)):
        return sum([_size(elem) for elem in data])
    elif isinstance(data, (str, bytes)):
        return len(data)
    elif isinstance(data, (int, float, complex)):
        return sys.getsizeof(data)
    return 0


def _count_true(data, filter_func):
    if isinstance(data, dict):
        return sum([_count_true(elem, filter_func) for elem in data.values()])
    elif isinstance(data, (list, tuple, set)):
        return sum([_count_true(elem, filter_func) for elem in data])
    elif isinstance(data, (int, float, complex, str, bytes)):
        return sum([filter_func(data)])
    elif data is None:
        return sum([filter_func(data)])
    return 0
