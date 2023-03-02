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
from graphsignal.uploader import Uploader

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

    @patch.object(Uploader, 'upload_signal')
    def test_shutdown_upload(self, mocked_upload_signal):
        with graphsignal.start_trace('test', options=graphsignal.TraceOptions(auto_sampling=False)):
            pass
        graphsignal.shutdown()

        signal = mocked_upload_signal.call_args[0][0]

        self.assertEqual(signal.deployment_name, 'd1')
        self.assertEqual(signal.endpoint_name, 'test')
        self.assertEqual(signal.signal_type, signals_pb2.SignalType.SNAPSHOT_SIGNAL)
