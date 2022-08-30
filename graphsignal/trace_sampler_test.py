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
            debug_mode=True)

    def tearDown(self):
        graphsignal.shutdown()

    def test_ensured(self):
        sampler = TraceSampler()
        for _ in range(TraceSampler.MAX_ENSURED_SPANS):
            self.assertTrue(sampler.lock(ensure=True))
            sampler.unlock()
        sampler.unlock()

    def test_predefined(self):
        sampler = TraceSampler()
        first_span = next(iter(sampler._span_filter))
        for idx in range(TraceSampler.PREDEFINED_SPANS[-1]):
            if idx + 1 in sampler._span_filter:
                self.assertTrue(sampler.lock())
                sampler.unlock()
            else:
                self.assertFalse(sampler.lock())
                sampler.unlock()
        self.assertFalse(sampler.lock())
        sampler.unlock()

    def test_interval(self):
        sampler = TraceSampler()
        sampler._span_filter = {}
        self.assertFalse(sampler.lock())
        sampler.unlock()
        self.assertFalse(sampler._interval_mode)

        # first interval
        sampler._start_ts = time.time() - TraceSampler.MIN_INTERVAL_SEC - 10
        self.assertTrue(sampler.lock())
        sampler.unlock()
        self.assertFalse(sampler.lock())
        sampler.unlock()
        self.assertTrue(sampler._interval_mode)

        # other intervals
        sampler._start_ts = time.time() - TraceSampler.MIN_INTERVAL_SEC - 10
        sampler._last_sample_ts = time.time() - TraceSampler.MIN_INTERVAL_SEC - 10
        self.assertTrue(sampler.lock())
        sampler.unlock()
        self.assertFalse(sampler.lock())
        sampler.unlock()

        # test ensured during interval
        self.assertFalse(sampler.lock())
        for _ in range(TraceSampler.MAX_ENSURED_SPANS):
            self.assertTrue(sampler.lock(ensure=True))
            sampler.unlock()
        self.assertFalse(sampler.lock())
        sampler.unlock()

    def test_concurrent(self):
        sampler = TraceSampler()
        self.assertTrue(sampler.lock(ensure=True))
        self.assertFalse(sampler.lock(ensure=True))
        self.assertFalse(sampler.lock(ensure=True))
        sampler.unlock()
        self.assertTrue(sampler.lock(ensure=True))
        sampler.unlock()

