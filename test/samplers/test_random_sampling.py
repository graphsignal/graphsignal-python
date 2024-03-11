import unittest
import logging
import sys
import time
from unittest.mock import patch, Mock

import graphsignal
from graphsignal.samplers.random_sampling import RandomSampler

logger = logging.getLogger('graphsignal')


class RandomSamplerTest(unittest.TestCase):
    def setUp(self):
        graphsignal.configure(
            api_key='k1',
            deployment='d1',
            upload_on_shutdown=False,
            debug_mode=True)

    def tearDown(self):
        graphsignal.shutdown()

    def test_sample(self):
        sampler = RandomSampler()

        for _ in range(10000):
            self.assertTrue(sampler.sample())

        self.assertFalse(sampler.sample())

        sampler._last_reset_ts = time.time() - 61

        self.assertTrue(sampler.sample())
