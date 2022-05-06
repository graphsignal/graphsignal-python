import unittest
import logging
import sys
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

    def test_default(self):
        scheduler = ProfileScheduler()
        for _ in range(next(iter(scheduler._step_filter)) - 1):
            scheduler.lock()
            scheduler.unlock()
        self.assertTrue(scheduler.lock())
        scheduler.unlock()

    def test_ensured(self):
        scheduler = ProfileScheduler()
        self.assertTrue(scheduler.lock(ensure=True))
        scheduler.unlock()
        self.assertFalse(scheduler.lock(ensure=False))
        scheduler.unlock()
        self.assertTrue(scheduler.lock(ensure=True))
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
