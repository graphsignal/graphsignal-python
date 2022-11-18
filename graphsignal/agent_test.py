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

    def test_create_signal(self):
        signal = graphsignal._agent.create_signal()
        self.assertTrue(signal.agent_info.version.major > 0 or signal.agent_info.version.minor > 0)

    @patch('time.time', return_value=1)
    def test_update_metric_store(self, mocked_time):
        store = graphsignal._agent.metric_store('ep1')

        store.inc_call_count(1, 1000 * 1e6)
        store.inc_call_count(1, 1000 * 1e6)
        store.inc_call_count(1, 1001 * 1e6)
        self.assertEqual(store.call_count.counter.buckets, {1: 0, 1000: 2, 1001: 1})
        
        store.inc_exception_count(1, 1003 * 1e6)
        store.inc_exception_count(1, 1003 * 1e6)
        store.inc_exception_count(1, 1004 * 1e6)
        self.assertEqual(store.exception_count.counter.buckets, {1: 0, 1003: 2, 1004: 1})

        store.inc_data_counter('d1', 'c1', 1, 1000 * 1e6)
        store.inc_data_counter('d1', 'c1', 1, 1000 * 1e6)
        store.inc_data_counter('d1', 'c1', 1, 1001 * 1e6)
        store.inc_data_counter('d1', 'c2', 1, 1001 * 1e6)
        self.assertEqual(store.data_counters[('d1', 'c1')].metric.counter.buckets, {1: 0, 1000: 2, 1001: 1})
        self.assertEqual(store.data_counters[('d1', 'c2')].metric.counter.buckets, {1: 0, 1001: 1})

        store.add_time(20)
        store.add_time(30)
        self.assertEqual(store.latency_us.reservoir.values, [20, 30])

