import unittest
import logging
import sys
import time
from unittest.mock import patch, Mock
import numpy as np
import torch

import graphsignal
from graphsignal.proto import signals_pb2
from graphsignal.data.torch_tensor import TorchTensorProfiler

logger = logging.getLogger('graphsignal')


class TorchTensorProfilerTest(unittest.TestCase):
    def setUp(self):
        if len(logger.handlers) == 0:
            logger.addHandler(logging.StreamHandler(sys.stdout))
        graphsignal.configure(
            api_key='k1',
            debug_mode=True)

    def tearDown(self):
        graphsignal.shutdown()

    def test_is_instance(self):
        profiler = TorchTensorProfiler()
        self.assertTrue(profiler.is_instance(torch.tensor([])))

    def test_get_size(self):
        profiler = TorchTensorProfiler()
        self.assertEqual(profiler.get_size(torch.tensor(np.asarray([[1, 2], [3, 4]]))), (4, 'elem'))

    def test_compute_stats(self):
        profiler = TorchTensorProfiler()
        ds = profiler.compute_stats(torch.tensor(np.asarray([[1, 2.0, 0, np.inf], [1, 0, 0, np.nan]])))
        self.assertEqual(ds.data_type, 'torch.Tensor')
        self.assertEqual(ds.size, 8)
        self.assertEqual(ds.shape, [2, 4])
        self.assertEqual(ds.num_nan, 1)
        self.assertEqual(ds.num_inf, 1)
        self.assertEqual(ds.num_zero, 3)
        self.assertEqual(ds.num_unique, 5)