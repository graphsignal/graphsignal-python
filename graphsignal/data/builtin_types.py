import logging
import sys
import json

import graphsignal
from graphsignal.proto import signals_pb2
from graphsignal.data.base_data_profiler import BaseDataProfiler, DataStats, DataSample

logger = logging.getLogger('graphsignal')


class BuiltInTypesProfiler(BaseDataProfiler):
    def __init__(self):
        super().__init__()

    def is_instance(self, data):
        return isinstance(data, (list, dict, tuple, set, str, int, float, complex)) or data is None

    def compute_stats(self, data):
        shape = None
        counts = {}
        if isinstance(data, bytes):
            counts['byte_count'] = len(data)
        elif isinstance(data, str):
            counts['char_count'] = len(data)
            if data == '':
                counts['empty_count'] = 1
        elif isinstance(data, (list, tuple, set)):
            if _is_array(data):
                shape = _shape(data)
                counts['element_count'] = _elems(data)
            else:
                counts['element_count'] = len(data)
        elif data is None:
            counts['null_count'] = 1
        return DataStats(type_name=type(data).__name__, shape=shape, counts=counts)

    def encode_sample(self, data):
        return DataSample(content_type='application/json', content_bytes=json.dumps(data).encode('utf-8'))


def _is_array(data):
    if isinstance(data, list):
        if len(data) > 0:
            if isinstance(data[0], (list, str, bytes, int, float, complex)):
                return True
        else:
            return True

    return False


def _shape(data):
    if isinstance(data, list):
        if len(data) == 0:
            return [0]
        else:
            return [len(data)] + _shape(data[0])
    return []


def _elems(data):
    if isinstance(data, list):
         return sum([_elems(elem) for elem in data])
    else:
        return 1
