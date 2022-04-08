import unittest
import logging
import sys
from unittest.mock import patch, Mock

import graphsignal
from graphsignal.span_counter import reset_span_stats, get_span_stats, update_span_stats

logger = logging.getLogger('graphsignal')


class SpanCounterTest(unittest.TestCase):
    def setUp(self):
        graphsignal.configure(
            api_key='k1',
            workload_name='w1',
            debug_mode=True)

    def tearDown(self):
        graphsignal.shutdown()

    def test_update(self):
        reset_span_stats()
        
        ss = get_span_stats(1)
        self.assertEqual(ss.count, 0)
        self.assertEqual(ss.total_time_us, 0)

        ss = update_span_stats(1, 100)
        ss = update_span_stats(1, 200)
        self.assertEqual(ss.count, 2)
        self.assertEqual(ss.total_time_us, 300)

        ss = get_span_stats(2)
        self.assertEqual(ss.count, 0)
        self.assertEqual(ss.total_time_us, 0)
