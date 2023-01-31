import unittest
import logging
import sys
import os
import time
from unittest.mock import patch, Mock
from google.protobuf.json_format import MessageToJson
import pprint

import graphsignal
from graphsignal.proto import signals_pb2

logger = logging.getLogger('graphsignal')

class AgentTest(unittest.TestCase):
    def setUp(self):
        graphsignal.configure(
            api_key='k1',
            deployment='d1',
            debug_mode=True)

    def tearDown(self):
        graphsignal.shutdown()

    def test_create_signal(self):
        signal = graphsignal._agent.create_signal()
        self.assertTrue(signal.agent_info.version.major > 0 or signal.agent_info.version.minor > 0)
