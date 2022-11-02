import unittest
import logging
import sys
import time
from unittest.mock import patch, Mock

import graphsignal
from graphsignal.proto import signals_pb2
from graphsignal.data.missing_value_detector import MissingValueDetector

logger = logging.getLogger('graphsignal')


class MissingValueDetectorTest(unittest.TestCase):
    def setUp(self):
        if len(logger.handlers) == 0:
            logger.addHandler(logging.StreamHandler(sys.stdout))
        graphsignal.configure(
            api_key='k1',
            debug_mode=True)

    def tearDown(self):
        graphsignal.shutdown()

    def test_detect(self):
        mvd = MissingValueDetector()
        self.assertTrue(mvd.detect('d1', {'null_count': 1}))
        self.assertFalse(mvd.detect('d1', {'null_count': 0}))
