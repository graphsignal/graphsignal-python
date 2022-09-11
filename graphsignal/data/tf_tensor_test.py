import unittest
import logging
import sys
import time
from unittest.mock import patch, Mock
import numpy as np
import tensorflow as tf

import graphsignal
from graphsignal.proto import signals_pb2
from graphsignal.data.tf_tensor import TFTensorProfiler

logger = logging.getLogger('graphsignal')


class TFTensorProfilerTest(unittest.TestCase):
    def setUp(self):
        if len(logger.handlers) == 0:
            logger.addHandler(logging.StreamHandler(sys.stdout))
        graphsignal.configure(
            api_key='k1',
            debug_mode=True)

    def tearDown(self):
        graphsignal.shutdown()

    def test_is_instance(self):
        profiler = TFTensorProfiler()
        self.assertTrue(profiler.is_instance(tf.constant([])))

    def test_get_size(self):
        profiler = TFTensorProfiler()
        self.assertEqual(profiler.get_size(tf.constant([[1, 2], [3, 4]])), (4, 'elem'))

    def test_compute_stats(self):
        profiler = TFTensorProfiler()
        ds = profiler.compute_stats(tf.constant(np.asarray([[1, 2.0, 0, np.inf], [1, 0, 0, np.nan]])))
        self.assertEqual(ds.data_type, 'tf.Tensor')
        self.assertEqual(ds.size, 8)
        self.assertEqual(ds.shape, [2, 4])
        self.assertEqual(ds.num_nan, 1)
        self.assertEqual(ds.num_inf, 1)
        self.assertEqual(ds.num_zero, 3)
