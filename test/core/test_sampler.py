import unittest
import time
from unittest.mock import patch, MagicMock

from graphsignal.core.sampler import TimeCoordinatedSampler


class TimeCoordinatedSamplerTest(unittest.TestCase):
    @patch('time.time_ns')
    def test_should_sample(self, mocked_time_ns):
        sampler = TimeCoordinatedSampler(sampling_rate=0.1) # every 10 seconds

        mocked_time_ns.return_value = 1765199953205064883
        self.assertTrue(sampler.should_sample())
        self.assertFalse(sampler.should_sample())

        mocked_time_ns.return_value = 1765199953205064883 + 10_000_000_000
        self.assertTrue(sampler.should_sample())

