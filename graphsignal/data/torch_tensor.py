import logging
import functools

import graphsignal
from graphsignal.proto import signals_pb2
from graphsignal.data.data_profiler import DataProfiler

logger = logging.getLogger('graphsignal')


class TorchTensorProfiler(DataProfiler):
    def __init__(self):
        super().__init__()

    def is_instance(self, data):
        torch = self.check_module('torch')
        return torch is not None and torch.is_tensor(data)

    def get_size(self, data):
        return (functools.reduce(lambda x, y: x*y, list(data.size())), 'elem')

    def compute_stats(self, data):
        torch = self.check_module('torch')
        data_stats = signals_pb2.DataStats()
        data_stats.data_type = 'torch.Tensor'
        data_stats.size, _ = self.get_size(data)
        data_stats.shape[:] = list(data.size())
        data_stats.num_nan = torch.count_nonzero(torch.isnan(data))
        data_stats.num_inf = torch.count_nonzero(torch.isinf(data))
        data_stats.num_zero = torch.count_nonzero(torch.eq(data, 0))
        data_stats.num_unique = len(torch.unique(data))
        return data_stats
