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
            deployment='d1',
            upload_on_shutdown=False,
            debug_mode=True)

    def tearDown(self):
        graphsignal.shutdown()

    def test_is_instance(self):
        profiler = NumpyNDArrayProfiler()
        self.assertTrue(profiler.is_instance(np.asarray([])))

    def test_compute_stats(self):
        profiler = NumpyNDArrayProfiler()
        stats = profiler.compute_stats(np.asarray([[-1, 2.0, 0, np.inf], [1, 0, 0, np.nan]]))
        self.assertEqual(
            stats.counts, 
            {'element_count': 8, 
            'byte_count': 64,
            'nan_count': 1, 
            'inf_count': 1, 
            'zero_count': 3,
            'negative_count': 1,
            'positive_count': 3})
        self.assertEqual(stats.type_name, 'numpy.ndarray')
        self.assertEqual(stats.shape, [2, 4])

    def test_encode_sample(self):
        profiler = NumpyNDArrayProfiler()
        preview = profiler.encode_sample(np.asarray([[-1, 2.0, 0, np.inf]]))
        self.assertEqual(preview.content_type, 'application/json')
        self.assertEqual(preview.content_bytes, b'[[-1.0, 2.0, 0.0, Infinity]]')
