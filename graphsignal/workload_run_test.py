import unittest
import logging
import sys
import time
from unittest.mock import patch, Mock

import graphsignal
from graphsignal.workload_run import WorkloadRun

logger = logging.getLogger('graphsignal')


class WorkloadRunTest(unittest.TestCase):
    def setUp(self):
        graphsignal.configure(
            api_key='k1',
            workload_name='w1',
            debug_mode=True)

    def tearDown(self):
        graphsignal.shutdown()

    def test_update_inference_stats(self):
        wr = WorkloadRun()

        wr.update_inference_stats(10, batch_size=128)
        wr.update_inference_stats(20, batch_size=128)
        self.assertEqual(wr.inference_count, 2)
        self.assertEqual(wr.total_time_us, 30)
        self.assertEqual(wr.sample_count, 2 * 128)
