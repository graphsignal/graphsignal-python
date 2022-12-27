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
            debug_mode=True)

    def tearDown(self):
        graphsignal.shutdown()

    def test_group(self):
        limit = 2
        sampler = TraceSampler()
        for _ in range(limit):
            self.assertTrue(sampler.lock('g1', limit_per_interval=limit))
            sampler.unlock()
        self.assertFalse(sampler.lock('g1', limit_per_interval=limit))
        sampler.unlock()

    def test_include_trace_idx(self):
        sampler = TraceSampler()
        trace_idx = {1, 4, 10}
        for idx in range(list(trace_idx)[-1]):
            if idx + 1 in trace_idx:
                self.assertTrue(sampler.lock('g1', limit_per_interval=0, include_trace_idx=trace_idx))
                sampler.unlock()
            else:
                self.assertFalse(sampler.lock('g1', limit_per_interval=0, include_trace_idx=trace_idx))
                sampler.unlock()
        self.assertFalse(sampler.lock('g1', limit_per_interval=0, include_trace_idx=trace_idx))
        sampler.unlock()

    def test_concurrent(self):
        sampler = TraceSampler()
        self.assertTrue(sampler.lock('g1'))
        self.assertFalse(sampler.lock('g1'))
        self.assertFalse(sampler.lock('g2'))
        sampler.unlock()
        self.assertTrue(sampler.lock('g2'))
        sampler.unlock()

    def test_reset(self):
        sampler = TraceSampler()

        self.assertTrue(sampler.lock('g1', limit_per_interval=1))
        sampler.unlock()
        self.assertFalse(sampler.lock('g1', limit_per_interval=1))
        sampler.unlock()

        sampler._last_reset_ts = time.time() - TraceSampler.MIN_INTERVAL_SEC - 10

        self.assertTrue(sampler.lock('g1', limit_per_interval=1))
        sampler.unlock()
        self.assertFalse(sampler.lock('g1', limit_per_interval=1))
        sampler.unlock()
