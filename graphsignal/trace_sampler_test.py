import unittest
import logging
import sys
import time
from unittest.mock import patch, Mock

import graphsignal
from graphsignal.trace_sampler import TraceSampler

logger = logging.getLogger('graphsignal')


class TraceSamplerTest(unittest.TestCase):
    def setUp(self):
        graphsignal.configure(
            api_key='k1',
            deployment='d1',
            upload_on_shutdown=False,
            debug_mode=True)

    def tearDown(self):
        graphsignal.shutdown()

    def test_group(self):
        limit = 2
        sampler = TraceSampler()
        for _ in range(limit):
            self.assertTrue(sampler.sample('g1', limit_per_interval=limit, limit_after=0))
        self.assertFalse(sampler.sample('g1', limit_per_interval=limit, limit_after=0))

    def test_limit_after(self):
        sampler = TraceSampler()
        for _ in range(10):
            self.assertTrue(sampler.sample('g1', limit_per_interval=0, limit_after=10))
        self.assertFalse(sampler.sample('g1', limit_per_interval=0, limit_after=10))

    def test_reset(self):
        sampler = TraceSampler()

        self.assertTrue(sampler.sample('g1', limit_per_interval=1, limit_after=0))
        self.assertFalse(sampler.sample('g1', limit_per_interval=1, limit_after=0))

        sampler._last_reset_ts = time.time() - TraceSampler.MIN_INTERVAL_SEC - 10

        self.assertTrue(sampler.sample('g1', limit_per_interval=1, limit_after=0))
        self.assertFalse(sampler.sample('g1', limit_per_interval=1, limit_after=0))
