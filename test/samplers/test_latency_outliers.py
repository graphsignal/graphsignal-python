import unittest
import logging
import sys
import time
from unittest.mock import patch, Mock
import random

import graphsignal
from graphsignal.samplers.latency_outliers import LatencyOutlierSampler

logger = logging.getLogger('graphsignal')

class LatencyOutlierSamplerTest(unittest.TestCase):
    def setUp(self):
        graphsignal.configure(
            api_key='k1',
            deployment='d1',
            upload_on_shutdown=False,
            debug_mode=True)

    def tearDown(self):
        graphsignal.shutdown()

    def test_sample(self):
        sampler = LatencyOutlierSampler()

        for i in range(500):
            val = random.randint(100, 200)
            self.assertFalse(sampler.sample(val))
            sampler.update(val)
        self.assertTrue(sampler.sample(1000))