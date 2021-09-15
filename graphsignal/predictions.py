import logging
import time

logger = logging.getLogger('graphsignal')


class Prediction(object):
    __slots__ = [
        'input_data',
        'input_columns',
        'output_data',
        'output_columns'
    ]

    def __init__(
            self,
            input_data=None,
            input_columns=None,
            output_data=None,
            output_columns=None):
        self.input_data = input_data
        self.input_columns = input_columns
        self.output_data = output_data
        self.output_columns = output_columns
