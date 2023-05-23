import logging
import json

import graphsignal
from graphsignal.proto import signals_pb2
from graphsignal.data.base_data_profiler import BaseDataProfiler, DataStats, DataSample

logger = logging.getLogger('graphsignal')


class NumpyNDArrayProfiler(BaseDataProfiler):
    def __init__(self):
        super().__init__()

    def is_instance(self, data):
        np = self.check_module('numpy')
        return np is not None and isinstance(data, np.ndarray)

    def compute_stats(self, data):
        np = self.check_module('numpy')
        counts = {}
        counts['element_count'] = data.size
        counts['byte_count'] = data.nbytes
        counts['null_count'] = np.count_nonzero(data == None)
        if np.issubdtype(data.dtype, np.number):
            counts['nan_count'] = np.count_nonzero(np.isnan(data))
            counts['inf_count'] = np.count_nonzero(np.isinf(data))
        counts['zero_count'] = np.count_nonzero(data == 0)
        counts['negative_count'] = np.count_nonzero(data < 0)
        counts['positive_count'] = np.count_nonzero(data > 0)
        return DataStats(type_name='numpy.ndarray', shape=list(data.shape), counts=counts)

    def encode_sample(self, data):
        return DataSample(content_type='application/json', content_bytes=json.dumps(data.tolist()).encode('utf-8'))
