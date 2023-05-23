import logging
import importlib
import functools

import graphsignal
from graphsignal.data.builtin_types import BuiltInTypesProfiler
from graphsignal.data.numpy_ndarray import NumpyNDArrayProfiler
from graphsignal.data.tf_tensor import TFTensorProfiler
from graphsignal.data.torch_tensor import TorchTensorProfiler
from graphsignal.data.base_data_profiler import DataStats


logger = logging.getLogger('graphsignal')


data_profilers = [
    NumpyNDArrayProfiler(),
    TorchTensorProfiler(),
    TFTensorProfiler(),
    BuiltInTypesProfiler()
]


def compute_data_stats(data):
    for dp in data_profilers:
        if dp.is_instance(data):
            return dp.compute_stats(data)


def encode_data_sample(data):
    for dp in data_profilers:
        if dp.is_instance(data):
            return dp.encode_sample(data)

    return None
