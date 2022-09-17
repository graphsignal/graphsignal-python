import logging
import functools

import graphsignal
from graphsignal.proto import signals_pb2
from graphsignal.data.data_profiler import DataProfiler, add_counts

logger = logging.getLogger('graphsignal')


class TorchTensorProfiler(DataProfiler):
    def __init__(self):
        super().__init__()

    def is_instance(self, data):
        torch = self.check_module('torch')
        return torch is not None and torch.is_tensor(data)

    def compute_counts(self, data):
        torch = self.check_module('torch')
        counts = {}
        counts['element_count'] = functools.reduce(lambda x, y: x*y, list(data.size()))
        counts['nan_count'] = int(torch.count_nonzero(torch.isnan(data)))
        counts['inf_count'] = int(torch.count_nonzero(torch.isinf(data)))
        counts['zero_count'] = int(torch.count_nonzero(torch.eq(data, 0)))
        return counts

    def build_stats(self, data):
        counts = self.compute_counts(data)
        data_stats = signals_pb2.DataStats()
        data_stats.data_type = 'torch.Tensor'
        data_stats.shape[:] = list(data.size())
        add_counts(data_stats, counts)
        return data_stats