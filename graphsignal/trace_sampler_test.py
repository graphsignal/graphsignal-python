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
        sampler = TraceSampler()
        for _ in range(sampler.EXTRA_SAMPLES):
            self.assertTrue(sampler.sample('g1'))

        for _ in range(1000):
            sampler.sample('g1')

        self.assertFalse(sampler.sample('g1'))
