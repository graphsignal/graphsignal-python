import unittest
import logging
import sys
import time
from unittest.mock import patch, Mock

import graphsignal
from graphsignal.profile_scheduler import ProfileScheduler

logger = logging.getLogger('graphsignal')


class ProfileSchedulerTest(unittest.TestCase):
    def setUp(self):
        graphsignal.configure(
            api_key='k1',
            workload_name='w1',
            debug_mode=True)

    def tearDown(self):
        graphsignal.shutdown()

    def test_ensured(self):
        scheduler = ProfileScheduler()
        for _ in range(ProfileScheduler.MAX_ENSURED_SPANS):
            self.assertTrue(scheduler.lock(ensure=True))
            scheduler.unlock()
        scheduler.unlock()

    def test_predefined(self):
        scheduler = ProfileScheduler()
        first_span = next(iter(scheduler._span_filter))
        for _ in range(first_span - 1):
            self.assertFalse(scheduler.lock())
            scheduler.unlock()
        self.assertTrue(scheduler.lock())
        scheduler.unlock()

    def test_interval(self):
        scheduler = ProfileScheduler()
        scheduler._span_filter = {}
        scheduler._last_profiled_ts = time.time()
        scheduler._current_span = 5
        scheduler._last_profiled_span = 5
        for _ in range(ProfileScheduler.MIN_INTERVAL_SPANS):
            self.assertFalse(scheduler.lock())
            scheduler.unlock()
        scheduler._last_profiled_ts = time.time() - ProfileScheduler.MIN_INTERVAL_SEC - 1
        self.assertTrue(scheduler.lock())
        scheduler.unlock()

    def test_concurrent(self):
        scheduler = ProfileScheduler()
        self.assertTrue(scheduler.lock(ensure=True))
        self.assertFalse(scheduler.lock(ensure=True))
        self.assertFalse(scheduler.lock(ensure=True))
        scheduler.unlock()
        self.assertTrue(scheduler.lock(ensure=True))
        scheduler.unlock()

