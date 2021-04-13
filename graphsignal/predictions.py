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
        'timestamp'
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
            timestamp=None):
        self.input_data = input_data
        self.input_type = input_type
        self.output_data = output_data
        self.output_type = output_type
        self.context_data = context_data
        self.timestamp = timestamp if timestamp is not None else _now()

    @staticmethod
    def data_type(data_type_name):
        return Prediction._data_type_map[data_type_name]


def _now():
    return int(time.time())
