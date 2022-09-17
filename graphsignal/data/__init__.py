import logging
import importlib
import functools

import graphsignal
from graphsignal.data.builtin_types import BuiltInTypesProfiler
from graphsignal.data.numpy_ndarray import NumpyNDArrayProfiler
from graphsignal.data.tf_tensor import TFTensorProfiler
from graphsignal.data.torch_tensor import TorchTensorProfiler

logger = logging.getLogger('graphsignal')


data_profilers = [
    BuiltInTypesProfiler(),
    NumpyNDArrayProfiler(),
    TFTensorProfiler(),
    TorchTensorProfiler()
]


def compute_counts(data):
    for dp in data_profilers:
        if dp.is_instance(data):
            return dp.compute_counts(data)


def build_stats(data):
    for dp in data_profilers:
        if dp.is_instance(data):
            return dp.build_stats(data)
