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
            deployment='d1',
            upload_on_shutdown=False,
            debug_mode=True)

    def tearDown(self):
        graphsignal.shutdown()

    def test_is_instance(self):
        profiler = BuiltInTypesProfiler()
        self.assertTrue(profiler.is_instance([]))

    def test_compute_stats(self):
        profiler = BuiltInTypesProfiler()
        stats = profiler.compute_stats([[0, -1, 1, 2.0, 3, float('nan'), float('inf')], [0, None, ''],[6,7]])
        self.assertEqual(
            stats.counts,
            {'byte_count': 260,
             'element_count': 12,
             'inf_count': 1,
             'nan_count': 1,
             'null_count': 1,
             'zero_count': 2,
             'empty_count': 1,
             'negative_count': 1,
             'positive_count': 6})
        self.assertEqual(stats.type_name, 'list')
        self.assertEqual(stats.shape, [3, 7])

    def test_compute_stats_none(self):
        profiler = BuiltInTypesProfiler()
        stats = profiler.compute_stats(None)
        self.assertEqual(
            stats.counts, 
            {'element_count': 1,
             'null_count': 1})
        self.assertEqual(stats.type_name, 'NoneType')
