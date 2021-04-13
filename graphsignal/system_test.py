import unittest
import logging


from unittest.mock import patch, Mock

from graphsignal import system

logger = logging.getLogger('graphsignal')


class SystemTest(unittest.TestCase):
    def setUp(self):
        logger.setLevel(logging.DEBUG)

    def tearDown(self):
        pass

    def test_cpu_time(self):
        self.assertTrue(system.cpu_time() > 0)

    def test_vm_rss(self):
        self.assertTrue(system.vm_rss() > 0)

    def test_vm_size(self):
        self.assertTrue(system.vm_size() > 0)
