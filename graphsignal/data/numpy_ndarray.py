import logging

import graphsignal
from graphsignal.proto import signals_pb2
from graphsignal.data.data_profiler import DataProfiler

logger = logging.getLogger('graphsignal')


class NumpyNDArrayProfiler(DataProfiler):
    def __init__(self):
        super().__init__()

    def is_instance(self, data):
        np = self.check_module('numpy')        
        return np is not None and isinstance(data, np.ndarray)

    def get_size(self, data):
        return (data.size, 'elem')

    def compute_stats(self, data):
        np = self.check_module('numpy')
        data_stats = signals_pb2.DataStats()
        data_stats.data_type = 'numpy.ndarray'
        data_stats.size, _ = self.get_size(data)
        data_stats.shape[:] = list(data.shape)
        data_stats.num_null = np.count_nonzero(data == None)
        if np.issubdtype(data.dtype, np.number):
            data_stats.num_nan = np.count_nonzero(np.isnan(data))
            data_stats.num_inf = np.count_nonzero(np.isinf(data))
        data_stats.num_zero = np.count_nonzero(data == 0)
        data_stats.num_unique = len(np.unique(data))
        return data_stats
