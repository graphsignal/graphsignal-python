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
            deployment='d1',
            upload_on_shutdown=False,
            debug_mode=True)

    def tearDown(self):
        graphsignal.shutdown()

    def test_is_instance(self):
        profiler = TorchTensorProfiler()
        self.assertTrue(profiler.is_instance(torch.tensor([])))

    def test_compute_stats(self):
        profiler = TorchTensorProfiler()
        stats = profiler.compute_stats(torch.tensor(np.asarray([[-1, 2.0, 0, np.inf], [1, 0, 0, np.nan]])))
        self.assertEqual(
            stats.counts, 
            {'element_count': 8,
             'byte_count': 64,
             'nan_count': 1,
             'inf_count': 1,
             'zero_count': 3,
             'negative_count': 1,
             'positive_count': 3,})
        self.assertEqual(stats.type_name, 'torch.Tensor')
        self.assertEqual(stats.shape, [2, 4])
