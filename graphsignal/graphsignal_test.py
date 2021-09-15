import unittest
import logging
import sys
from unittest.mock import patch, Mock

import graphsignal

logger = logging.getLogger('graphsignal')


class GraphsignalTest(unittest.TestCase):
    def setUp(self):
        if len(logger.handlers) == 0:
            logger.addHandler(logging.StreamHandler(sys.stdout))
        graphsignal.configure(api_key='k1', debug_mode=True)

    def tearDown(self):
        graphsignal.shutdown()

    def test_configure(self):
        self.assertEqual(graphsignal._get_config().api_key, 'k1')
        self.assertEqual(graphsignal._get_config().debug_mode, True)
