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
        return True

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
        data_dict = _obj_to_dict(data)
        return DataSample(
            content_type='application/json', 
            content_bytes=json.dumps(data_dict).encode('utf-8'))


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


def _obj_to_dict(obj, level=0):
    if level >= 10:
        return
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    elif isinstance(obj, dict):
        return {k: _obj_to_dict(v, level=level+1) for k, v in obj.items()}
    elif isinstance(obj, (list, set, tuple)):
        return [_obj_to_dict(e, level=level+1) for e in obj]
    elif hasattr(obj, '__dict__'):
        return _obj_to_dict(vars(obj), level=level+1)
    else:
        return str(obj)