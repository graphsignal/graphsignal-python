import unittest
import logging
import sys
import time
from unittest.mock import patch, Mock
import random

import graphsignal
from graphsignal.detectors.latency_outlier_detector import LatencyOutlierDetector

logger = logging.getLogger('graphsignal')

class LatencyOutlierDetectorTest(unittest.TestCase):
    def setUp(self):
        graphsignal.configure(
            api_key='k1',
            deployment='d1',
            upload_on_shutdown=False,
            debug_mode=True)

    def tearDown(self):
        graphsignal.shutdown()

    def test_detect(self):
        detector = LatencyOutlierDetector()

        for i in range(500):
            val = random.randint(100, 200)
            self.assertFalse(detector.detect(val))
            detector.update(val)
        self.assertTrue(detector.detect(1000))