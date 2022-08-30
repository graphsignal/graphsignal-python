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

    @patch('time.time', return_value=1)
    def test_update_inference_stats(self, mocked_time):
        stats = graphsignal._agent.get_inference_stats('m1')

        stats.inc_inference_counter(1, 1000 * 1e6)
        stats.inc_inference_counter(1, 1000 * 1e6)
        stats.inc_inference_counter(1, 1001 * 1e6)
        self.assertEqual(stats.inference_counter, {1: 0, 1000: 2, 1001: 1})
        
        stats.inc_exception_counter(1, 1003 * 1e6)
        stats.inc_exception_counter(1, 1003 * 1e6)
        stats.inc_exception_counter(1, 1004 * 1e6)
        self.assertEqual(stats.exception_counter, {1: 0, 1003: 2, 1004: 1})

        stats.inc_extra_counter('c1', 1, 1000 * 1e6)
        stats.inc_extra_counter('c1', 1, 1000 * 1e6)
        stats.inc_extra_counter('c1', 1, 1001 * 1e6)
        stats.inc_extra_counter('c2', 1, 1001 * 1e6)
        self.assertEqual(stats.extra_counters['c1'], {1: 0, 1000: 2, 1001: 1})
        self.assertEqual(stats.extra_counters['c2'], {1: 0, 1001: 1})

        stats.add_time(20)
        stats.add_time(30)
        self.assertEqual(stats.time_reservoir_us, [20, 30])

