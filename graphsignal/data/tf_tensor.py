import logging
import functools

import graphsignal
from graphsignal.proto import signals_pb2
from graphsignal.data.data_profiler import DataProfiler

logger = logging.getLogger('graphsignal')


class TFTensorProfiler(DataProfiler):
    def __init__(self):
        super().__init__()

    def is_instance(self, data):
        tf = self.check_module('tensorflow')
        return tf is not None and tf.is_tensor(data)

    def get_size(self, data):
        tf = self.check_module('tensorflow')
        return (functools.reduce(lambda x, y: x*y, list(data.get_shape())), 'elem')

    def compute_stats(self, data):
        tf = self.check_module('tensorflow')
        data_stats = signals_pb2.DataStats()
        data_stats.data_type = 'tf.Tensor'
        data_stats.size, _ = self.get_size(data)
        data_stats.shape[:] = list(data.get_shape())
        data_stats.num_nan = tf.math.count_nonzero(tf.math.is_nan(data))
        data_stats.num_inf = tf.math.count_nonzero(tf.math.is_inf(data))
        data_stats.num_zero = tf.math.count_nonzero(tf.math.equal(data, 0))
        return data_stats
