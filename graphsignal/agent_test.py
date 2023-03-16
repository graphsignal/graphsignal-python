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

    def test_create_trace_proto(self):
        proto = graphsignal._agent.create_trace_proto()
        self.assertTrue(proto.agent_info.version.major > 0 or proto.agent_info.version.minor > 0)

    @patch.object(Uploader, 'upload_metric')
    def test_shutdown_upload(self, mocked_upload_metric):
        graphsignal._agent.metric_store().set_gauge(scope='s1', name='n1', tags={}, value=1, update_ts=1)
        graphsignal.shutdown()

        proto = mocked_upload_metric.call_args[0][0]

        self.assertEqual(proto.scope, 's1')
        self.assertEqual(proto.name, 'n1')
