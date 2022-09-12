import unittest
import logging
import sys
import os
import time
from unittest.mock import patch, Mock

import graphsignal

logger = logging.getLogger('graphsignal')


class AgentTest(unittest.TestCase):
    def setUp(self):
        graphsignal.configure(
            api_key='k1',
            debug_mode=True)

    def tearDown(self):
        graphsignal.shutdown()

    def test_tracer_default(self):
        tracer = graphsignal._agent.tracer()
        self.assertIsNotNone(tracer)
        self.assertTrue(isinstance(tracer.profiler(), graphsignal.profilers.python.PythonProfiler))

    def test_tracer_python(self):
        tracer = graphsignal._agent.tracer(with_profiler='python')
        self.assertIsNotNone(tracer)
        self.assertTrue(isinstance(tracer.profiler(), graphsignal.profilers.python.PythonProfiler))

    def test_tracer_disabled(self):
        tracer = graphsignal._agent.tracer(with_profiler=False)
        self.assertIsNotNone(tracer)
        self.assertIsNone(tracer.profiler())

    @patch('time.time', return_value=1)
    def test_update_span_stats(self, mocked_time):
        stats = graphsignal._agent.get_span_stats('m1')

        stats.inc_call_counter(1, 1000 * 1e6)
        stats.inc_call_counter(1, 1000 * 1e6)
        stats.inc_call_counter(1, 1001 * 1e6)
        self.assertEqual(stats.call_counter.buckets_sec, {1: 0, 1000: 2, 1001: 1})
        
        stats.inc_exception_counter(1, 1003 * 1e6)
        stats.inc_exception_counter(1, 1003 * 1e6)
        stats.inc_exception_counter(1, 1004 * 1e6)
        self.assertEqual(stats.exception_counter.buckets_sec, {1: 0, 1003: 2, 1004: 1})

        stats.inc_data_counter('c1', 1, 'elem', 1000 * 1e6)
        stats.inc_data_counter('c1', 1, 'elem', 1000 * 1e6)
        stats.inc_data_counter('c1', 1, 'elem', 1001 * 1e6)
        stats.inc_data_counter('c2', 1, 'elem', 1001 * 1e6)
        self.assertEqual(stats.data_counters['c1'].buckets_sec, {1: 0, 1000: 2, 1001: 1})
        self.assertEqual(stats.data_counters['c1'].unit, 'elem')
        self.assertEqual(stats.data_counters['c2'].buckets_sec, {1: 0, 1001: 1})
        self.assertEqual(stats.data_counters['c2'].unit, 'elem')

        stats.add_time(20)
        stats.add_time(30)
        self.assertEqual(stats.time_reservoir_us, [20, 30])

