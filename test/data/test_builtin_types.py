import unittest
import logging
import sys
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
        class TestObject:
            pass

        profiler = BuiltInTypesProfiler()
        stats = profiler.compute_stats(b'text')
        self.assertEqual(
            stats.counts,
            {'byte_count': 4})
        self.assertEqual(stats.type_name, 'bytes')

        profiler = BuiltInTypesProfiler()
        stats = profiler.compute_stats('textstr')
        self.assertEqual(
            stats.counts,
            {'char_count': 7})
        self.assertEqual(stats.type_name, 'str')

        profiler = BuiltInTypesProfiler()
        stats = profiler.compute_stats([1, {}])
        self.assertEqual(
            stats.counts,
            {'element_count': 2})
        self.assertEqual(stats.type_name, 'list')

        profiler = BuiltInTypesProfiler()
        stats = profiler.compute_stats([[1, 2], [3, 4], [5, 6]])
        self.assertEqual(
            stats.counts,
            {'element_count': 6})
        self.assertEqual(stats.type_name, 'list')
        self.assertEqual(stats.shape, [3, 2])

        profiler = BuiltInTypesProfiler()
        stats = profiler.compute_stats(None)
        self.assertEqual(
            stats.counts, 
            {'null_count': 1})
        self.assertEqual(stats.type_name, 'NoneType')

    def test_encode_sample(self):
        profiler = BuiltInTypesProfiler()
        preview = profiler.encode_sample(['text\n', 2.0, float('nan')])
        self.assertEqual(preview.content_type, 'application/json')
        self.assertEqual(preview.content_bytes, b'["text\\n", 2.0, NaN]')
