import unittest
import logging
import sys
import os
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


    def test_env_tags(self):
        os.environ["GRAPHSIGNAL_TAGS"] = "t1,t 2"

        wr = WorkloadRun()

        self.assertEqual(wr.tags, {
            't1': True,
            't 2': True
        })

    def test_env_params(self):
        os.environ["GRAPHSIGNAL_PARAMS"] = "p1: v1, p2 : 2 "

        wr = WorkloadRun()

        self.assertEqual(wr.params, {
            'p1': 'v1',
            'p2': '2'
        })

    def test_update_inference_stats(self):
        wr = WorkloadRun()

        wr.update_inference_stats(10, batch_size=128)
        wr.update_inference_stats(30, batch_size=128)
        self.assertEqual(wr.inference_stats.inference_time_p95_us(), 30)
        self.assertEqual(wr.inference_stats.inference_time_avg_us(), 20)
        self.assertEqual(wr.inference_stats.inference_rate(), 49999.99999999999)
        self.assertEqual(wr.inference_stats.sample_rate(), 6399999.999999999)

    def test_inc_total_inference_count(self):
        wr = WorkloadRun()

        wr.inc_total_inference_count()
        wr.inc_total_inference_count()
        self.assertEqual(wr.total_inference_count, 2)
