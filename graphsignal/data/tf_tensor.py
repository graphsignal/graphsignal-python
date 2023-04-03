import logging
import functools

import graphsignal
from graphsignal.proto import signals_pb2
from graphsignal.data.base_data_profiler import BaseDataProfiler, DataStats

logger = logging.getLogger('graphsignal')


class TFTensorProfiler(BaseDataProfiler):
    def __init__(self):
        super().__init__()

    def is_instance(self, data):
        tf = self.check_module('tensorflow')
        return tf is not None and tf.is_tensor(data)

    def compute_stats(self, data):
        tf = self.check_module('tensorflow')
        counts = {}
        counts['element_count'] = functools.reduce(lambda x, y: x*y, list(data.get_shape()))
        counts['byte_count'] = counts['element_count'] * data.dtype.size
        counts['nan_count'] = int(tf.math.count_nonzero(tf.math.is_nan(data)))
        counts['inf_count'] = int(tf.math.count_nonzero(tf.math.is_inf(data)))
        counts['zero_count'] = int(tf.math.count_nonzero(tf.math.equal(data, 0)))
        counts['negative_count'] = int(tf.math.count_nonzero(tf.math.less(data, 0)))
        counts['positive_count'] = int(tf.math.count_nonzero(tf.math.greater(data, 0)))
        return DataStats(type_name='tf.Tensor', shape=list(data.get_shape()), counts=counts)

    def encode_sample(self, data):
        return None
