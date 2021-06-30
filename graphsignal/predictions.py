import logging
import time

logger = logging.getLogger('graphsignal')


class Prediction(object):
    __slots__ = [
        'input_data',
        'output_data',
        'timestamp'
    ]

    def __init__(
            self,
            input_data=None,
            output_data=None,
            timestamp=None):
        self.input_data = input_data
        self.output_data = output_data
        self.timestamp = timestamp if timestamp is not None else _now()


class DataWindow(object):
    __slots__ = [
        'data',
        'timestamp'
    ]

    def __init__(
            self,
            data=None,
            timestamp=None):
        self.data = data
        self.timestamp = timestamp

    def size(self):
        return self.data.shape[0]


def _now():
    return int(time.time())
