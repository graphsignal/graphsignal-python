import unittest
import logging
import sys
from unittest.mock import patch, Mock

import graphsignal
from graphsignal.step_counter import reset_step_stats, get_step_stats, update_step_stats

logger = logging.getLogger('graphsignal')


class StepCounterTest(unittest.TestCase):
    def setUp(self):
        graphsignal.configure(
            api_key='k1',
            workload_name='w1',
            debug_mode=True)

    def tearDown(self):
        graphsignal.shutdown()

    def test_update(self):
        reset_step_stats()
        
        ss = get_step_stats(1)
        self.assertEqual(ss.count, 0)
        self.assertEqual(ss.total_time_us, 0)

        ss = update_step_stats(1, 100)
        ss = update_step_stats(1, 200)
        self.assertEqual(ss.count, 2)
        self.assertEqual(ss.total_time_us, 300)

        ss = get_step_stats(2)
        self.assertEqual(ss.count, 0)
        self.assertEqual(ss.total_time_us, 0)
