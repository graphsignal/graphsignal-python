import unittest
import logging
import sys
import time
from unittest.mock import patch, Mock

import graphsignal
from graphsignal.profile_scheduler import ProfileScheduler, select_scheduler

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
        for _ in range(ProfileScheduler.MAX_ENSURED_STEPS):
            self.assertTrue(scheduler.lock(ensure=True))
            scheduler.unlock()
        scheduler.unlock()

    def test_predefined(self):
        scheduler = ProfileScheduler()
        first_step = next(iter(scheduler._step_filter))
        for _ in range(first_step - 1):
            self.assertFalse(scheduler.lock())
            scheduler.unlock()
        self.assertTrue(scheduler.lock())
        scheduler.unlock()

    def test_interval(self):
        scheduler = ProfileScheduler()
        scheduler._step_filter = {}
        scheduler._last_profiled_ts = time.time()
        scheduler._current_step = 5
        scheduler._last_profiled_step = 5
        for _ in range(ProfileScheduler.MIN_INTERVAL_STEPS):
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

    def test_select_scheduler(self):
        s1 = select_scheduler(None)
        s2 = select_scheduler(None)
        self.assertTrue(s1 == s2)

        s1 = select_scheduler('a')
        s2 = select_scheduler('a')
        self.assertTrue(s1 == s2)

        s1 = select_scheduler('b')
        s2 = select_scheduler('c')
        self.assertTrue(s1 != s2)

        for i in range(10):
            select_scheduler(str(i))

        sX = select_scheduler('x')
        self.assertIsNotNone(sX)
