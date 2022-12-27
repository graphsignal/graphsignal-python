import logging
import functools

import graphsignal
from graphsignal.proto import signals_pb2
from graphsignal.data.data_profiler import DataProfiler, add_counts

logger = logging.getLogger('graphsignal')


class TFTensorProfiler(DataProfiler):
    def __init__(self):
        super().__init__()

    def is_instance(self, data):
        tf = self.check_module('tensorflow')
        return tf is not None and tf.is_tensor(data)

    def compute_counts(self, data):
        tf = self.check_module('tensorflow')
        counts = {}
        counts['element_count'] = functools.reduce(lambda x, y: x*y, list(data.get_shape()))
        counts['byte_count'] = counts['element_count'] * data.dtype.size
        counts['nan_count'] = int(tf.math.count_nonzero(tf.math.is_nan(data)))
        counts['inf_count'] = int(tf.math.count_nonzero(tf.math.is_inf(data)))
        counts['zero_count'] = int(tf.math.count_nonzero(tf.math.equal(data, 0)))
        counts['negative_count'] = int(tf.math.count_nonzero(tf.math.less(data, 0)))
        counts['positive_count'] = int(tf.math.count_nonzero(tf.math.greater(data, 0)))
        return counts

    def build_stats(self, data):
        counts = self.compute_counts(data)
        data_stats = signals_pb2.DataStats()
        data_stats.data_type = 'tf.Tensor'
        data_stats.shape[:] = list(data.get_shape())
        add_counts(data_stats, counts)
        return data_stats
