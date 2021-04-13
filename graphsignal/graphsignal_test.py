import unittest
import logging
from unittest.mock import patch, Mock

import graphsignal

logger = logging.getLogger('graphsignal')


class GraphsignalTest(unittest.TestCase):
    def setUp(self):
        graphsignal.configure(api_key='k1', debug_mode=True)

    def tearDown(self):
        graphsignal.shutdown()

    def test_configure(self):
        self.assertEqual(graphsignal._get_config().api_key, 'k1')
        self.assertEqual(graphsignal._get_config().debug_mode, True)
        self.assertEqual(graphsignal._get_config().log_instances, True)
