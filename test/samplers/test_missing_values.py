import unittest
import logging
import sys
import time
from unittest.mock import patch, Mock

import graphsignal
from graphsignal.proto import signals_pb2
from graphsignal.samplers.missing_values import MissingValueSampler

logger = logging.getLogger('graphsignal')


class MissingValueSamplerTest(unittest.TestCase):
    def setUp(self):
        if len(logger.handlers) == 0:
            logger.addHandler(logging.StreamHandler(sys.stdout))
        graphsignal.configure(
            api_key='k1',
            deployment='d1',
            upload_on_shutdown=False,
            debug_mode=True)

    def tearDown(self):
        graphsignal.shutdown()

    def test_detect(self):
        mvs = MissingValueSampler()
        self.assertTrue(mvs.sample('d1', {'null_count': 1}))
        self.assertFalse(mvs.sample('d1', {'null_count': 0}))
