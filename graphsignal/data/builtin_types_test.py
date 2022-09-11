import unittest
import logging
import sys
import time
from unittest.mock import patch, Mock

import graphsignal
from graphsignal.proto import signals_pb2
from graphsignal.data.builtin_types import BuiltInTypesProfiler

logger = logging.getLogger('graphsignal')


class BuiltInTypesProfilerTest(unittest.TestCase):
    def setUp(self):
        if len(logger.handlers) == 0:
            logger.addHandler(logging.StreamHandler(sys.stdout))
        graphsignal.configure(
            api_key='k1',
            debug_mode=True)

    def tearDown(self):
        graphsignal.shutdown()

    def test_is_instance(self):
        profiler = BuiltInTypesProfiler()
        self.assertTrue(profiler.is_instance([]))

    def test_get_size(self):
        profiler = BuiltInTypesProfiler()
        self.assertEqual(profiler.get_size(dict(a=[1, 2.0, 3], b=['444', '55'], c=set([6,7]))), (7, 'elem'))

    def test_get_size_str(self):
        profiler = BuiltInTypesProfiler()
        self.assertEqual(profiler.get_size('abc123'), (6, 'char'))

    def test_get_size_none(self):
        profiler = BuiltInTypesProfiler()
        self.assertEqual(profiler.get_size(None), (1, 'elem'))

    def test_compute_stats(self):
        profiler = BuiltInTypesProfiler()
        ds = profiler.compute_stats(dict(a=[0, 1, 1, 2.0, 3, float('nan'), float('inf')], b=[0, None, None], c=set([6,7])))
        self.assertEqual(ds.data_type, 'dict')
        self.assertEqual(ds.size, 12)
        self.assertEqual(ds.num_null, 2)
        self.assertEqual(ds.num_nan, 1)
        self.assertEqual(ds.num_inf, 1)
        self.assertEqual(ds.num_zero, 2)
        self.assertEqual(ds.num_unique, 9)

    def test_compute_stats_none(self):
        profiler = BuiltInTypesProfiler()
        ds = profiler.compute_stats(None)
        self.assertEqual(ds.data_type, 'NoneType')
        self.assertEqual(ds.size, 1)
        self.assertEqual(ds.num_null, 1)
        self.assertEqual(ds.num_nan, 0)
        self.assertEqual(ds.num_inf, 0)
        self.assertEqual(ds.num_zero, 0)
        self.assertEqual(ds.num_unique, 1)
