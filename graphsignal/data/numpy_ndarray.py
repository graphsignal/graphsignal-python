import logging

import graphsignal
from graphsignal.proto import signals_pb2
from graphsignal.data.data_profiler import DataProfiler, add_counts

logger = logging.getLogger('graphsignal')


class NumpyNDArrayProfiler(DataProfiler):
    def __init__(self):
        super().__init__()

    def is_instance(self, data):
        np = self.check_module('numpy')        
        return np is not None and isinstance(data, np.ndarray)

    def compute_counts(self, data):
        np = self.check_module('numpy')
        counts = {}
        counts['element_count'] = data.size
        counts['null_count'] = np.count_nonzero(data == None)
        if np.issubdtype(data.dtype, np.number):
            counts['nan_count'] = np.count_nonzero(np.isnan(data))
            counts['inf_count'] = np.count_nonzero(np.isinf(data))
        counts['zero_count'] = np.count_nonzero(data == 0)
        return counts

    def build_stats(self, data):
        counts = self.compute_counts(data)
        data_stats = signals_pb2.DataStats()
        data_stats.data_type = 'numpy.ndarray'
        data_stats.shape[:] = list(data.shape)
        add_counts(data_stats, counts)
        return data_stats