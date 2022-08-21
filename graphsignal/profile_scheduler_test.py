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
        for idx in range(ProfileScheduler.PREDEFINED_SPANS[-1]):
            if idx + 1 in scheduler._span_filter:
                self.assertTrue(scheduler.lock())
                scheduler.unlock()
            else:
                self.assertFalse(scheduler.lock())
                scheduler.unlock()
        self.assertFalse(scheduler.lock())
        scheduler.unlock()

    def test_interval(self):
        scheduler = ProfileScheduler()
        scheduler._span_filter = {}
        self.assertFalse(scheduler.lock())
        scheduler.unlock()
        self.assertFalse(scheduler._interval_mode)

        # first interval
        scheduler._start_ts = time.time() - ProfileScheduler.MIN_INTERVAL_SEC - 10
        self.assertTrue(scheduler.lock())
        scheduler.unlock()
        self.assertTrue(scheduler._interval_mode)

        # other intervals
        scheduler._start_ts = time.time() - ProfileScheduler.MIN_INTERVAL_SEC - 10
        scheduler._last_profile_ts = time.time() - ProfileScheduler.MIN_INTERVAL_SEC - 10
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

