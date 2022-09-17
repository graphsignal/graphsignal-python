import unittest
import logging
import sys
import time
from unittest.mock import patch, Mock
import numpy as np

import graphsignal
from graphsignal.proto import signals_pb2
from graphsignal.data.numpy_ndarray import NumpyNDArrayProfiler

logger = logging.getLogger('graphsignal')


class NumpyNDArrayProfilerTest(unittest.TestCase):
    def setUp(self):
        if len(logger.handlers) == 0:
            logger.addHandler(logging.StreamHandler(sys.stdout))
        graphsignal.configure(
            api_key='k1',
            debug_mode=True)

    def tearDown(self):
        graphsignal.shutdown()

    def test_is_instance(self):
        profiler = NumpyNDArrayProfiler()
        self.assertTrue(profiler.is_instance(np.asarray([])))

    def test_compute_counts(self):
        profiler = NumpyNDArrayProfiler()
        self.assertEqual(
            profiler.compute_counts(np.asarray([[1, 2.0, 0, np.inf], [1, 0, 0, np.nan]])), 
            {'element_count': 8, 'null_count': 0, 'nan_count': 1, 'inf_count': 1, 'zero_count': 3})

    def test_build_stats(self):
        profiler = NumpyNDArrayProfiler()
        ds = profiler.build_stats(np.asarray([[1.], [2]]))
        self.assertEqual(ds.data_type, 'numpy.ndarray')
        self.assertEqual(ds.shape, [2, 1])

