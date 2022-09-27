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

    def test_compute_counts(self):
        profiler = BuiltInTypesProfiler()
        self.assertEqual(
            profiler.compute_counts(dict(a=[0, -1, 1, 2.0, 3, float('nan'), float('inf')], b=[0, None, ''], c=set([6,7]))), 
            {'element_count': 12,
             'inf_count': 1,
             'nan_count': 1,
             'null_count': 1,
             'zero_count': 2,
             'empty_count': 1,
             'negative_count': 1,
             'positive_count': 6})

    def test_compute_counts_none(self):
        profiler = BuiltInTypesProfiler()
        self.assertEqual(
            profiler.compute_counts(None), 
            {'element_count': 1,
             'inf_count': 0,
             'nan_count': 0,
             'null_count': 1,
             'zero_count': 0,
             'empty_count': 0,
             'negative_count': 0,
             'positive_count': 0})

    def test_build_stats(self):
        profiler = BuiltInTypesProfiler()
        ds = profiler.build_stats([1, 2])
        self.assertEqual(ds.data_type, 'list')

    def test_build_stats_none(self):
        profiler = BuiltInTypesProfiler()
        ds = profiler.build_stats(None)
        self.assertEqual(ds.data_type, 'NoneType')