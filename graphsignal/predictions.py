import logging
import time

logger = logging.getLogger('graphsignal')


class Prediction(object):
    __slots__ = [
        'input_data',
        'input_type',
        'output_data',
        'output_type',
        'context_data',
        'timestamp',
        'ensure_sample'
    ]

    DATA_TYPE_TABULAR = 1
    DATA_TYPE_TEXT = 2
    DATA_TYPE_IMAGE = 3

    _data_type_map = {
        'tabular': DATA_TYPE_TABULAR,
        'text': DATA_TYPE_TEXT,
        'image': DATA_TYPE_IMAGE
    }

    def __init__(
            self,
            input_data=None,
            input_type=None,
            output_data=None,
            output_type=None,
            context_data=None,
            ensure_sample=False,
            timestamp=None):
        self.input_data = input_data
        self.input_type = input_type
        self.output_data = output_data
        self.output_type = output_type
        self.context_data = context_data
        self.ensure_sample = ensure_sample
        self.timestamp = timestamp if timestamp is not None else _now()

    @staticmethod
    def data_type(data_type_name):
        return Prediction._data_type_map[data_type_name]


class DataWindow(object):
    __slots__ = [
        'data',
        'ensure_sample',
        'timestamp'
    ]

    def __init__(
            self,
            data=None,
            ensure_sample=None,
            timestamp=None):
        self.data = data
        self.ensure_sample = ensure_sample
        self.timestamp = timestamp

    def size(self):
        return self.data.shape[0]


def _now():
    return int(time.time())
