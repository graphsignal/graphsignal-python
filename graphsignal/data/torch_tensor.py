import logging
import functools

import graphsignal
from graphsignal.proto import signals_pb2
from graphsignal.data.base_data_profiler import BaseDataProfiler, DataStats

logger = logging.getLogger('graphsignal')


class TorchTensorProfiler(BaseDataProfiler):
    def __init__(self):
        super().__init__()

    def is_instance(self, data):
        torch = self.check_module('torch')
        return torch is not None and torch.is_tensor(data)

    def compute_stats(self, data):
        torch = self.check_module('torch')
        counts = {}
        counts['element_count'] = data.nelement()
        counts['byte_count'] = counts['element_count'] * data.element_size()
        counts['nan_count'] = int(torch.count_nonzero(torch.isnan(data)))
        counts['inf_count'] = int(torch.count_nonzero(torch.isinf(data)))
        counts['zero_count'] = int(torch.count_nonzero(torch.eq(data, 0)))
        counts['negative_count'] = int(torch.count_nonzero(torch.lt(data, 0)))
        counts['positive_count'] = int(torch.count_nonzero(torch.gt(data, 0)))
        return DataStats(type_name='torch.Tensor', shape=list(data.size()), counts=counts)

    def encode_sample(self, data):
        return None
