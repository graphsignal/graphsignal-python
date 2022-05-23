import unittest
import logging
import sys
import time
from unittest.mock import patch, Mock

import graphsignal
from graphsignal.step_counter import reset_all_step_stats, get_step_stats, update_step_stats

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
        reset_all_step_stats()
        
        ss = get_step_stats(1)
        self.assertEqual(ss.step_count, 0)
        self.assertEqual(ss.total_time_us, 0)

        ss = update_step_stats(1, 10, effective_batch_size=128)
        ss = update_step_stats(1, 20, effective_batch_size=128)
        self.assertEqual(ss.step_count, 2)
        self.assertEqual(ss.total_time_us, 30)
        self.assertEqual(ss.sample_count, 2 * 128)

        ss = get_step_stats(2)
        self.assertEqual(ss.step_count, 0)
        self.assertEqual(ss.total_time_us, 0)
        self.assertEqual(ss.sample_count, 0)
