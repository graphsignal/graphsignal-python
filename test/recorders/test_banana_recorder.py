import unittest
import logging
import sys
import os
import json
import time
from unittest.mock import patch, Mock
from google.protobuf.json_format import MessageToJson
import pprint
import banana_dev as banana

import graphsignal
from graphsignal.proto import signals_pb2
from graphsignal.uploader import Uploader
from graphsignal.spans import DEFAULT_OPTIONS
from graphsignal.recorders.banana_recorder import BananaRecorder
from graphsignal.proto_utils import find_tag, find_param, find_data_sample

logger = logging.getLogger('graphsignal')


class BananaRecorderTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        if len(logger.handlers) == 0:
            logger.addHandler(logging.StreamHandler(sys.stdout))
        graphsignal.configure(
            api_key='k1',
            deployment='d1',
            upload_on_shutdown=False,
            debug_mode=True)

    async def asyncTearDown(self):
        graphsignal.shutdown()

    async def test_record(self):
        recorder = BananaRecorder()
        recorder.setup()
        proto = signals_pb2.Span()
        context = {}
        recorder.on_span_start(proto, context, DEFAULT_OPTIONS)
        recorder.on_span_stop(proto, context, DEFAULT_OPTIONS)
        recorder.on_span_read(proto, context, DEFAULT_OPTIONS)

        self.assertEqual(proto.libraries[0].name, 'Banana Python SDK')

    @patch.object(Uploader, 'upload_span')
    @patch.object(banana, 'run')
    async def test_trace_run(self, mocked_run, mocked_upload_span):
        # mocking overrides autoinstrumentation, reinstrument
        recorder = BananaRecorder()
        recorder.setup()

        mocked_run.return_value = {
                "id": "12345678-1234-1234-1234-123456789012", 
                "message": "success", 
                "created": 1649712752, 
                "apiVersion": "26 Nov 2021", 
                "modelOutputs": [
                    {
                        "caption": "a baseball player throwing a ball"
                    },
                    {
                        "caption": "a baseball player throwing a bat"
                    }                    
                ]
            }

        response = banana.run(
            "api-key-1", 
            "model-key-1",
            ["test input 1", "test input 2"])

        recorder.shutdown()

        proto = mocked_upload_span.call_args[0][0]

        self.assertEqual(proto.libraries[0].name, 'Banana Python SDK')

        self.assertEqual(find_tag(proto, 'component'), 'Model')
        self.assertEqual(find_tag(proto, 'operation'), 'banana.run')
        self.assertEqual(find_tag(proto, 'endpoint'), 'https://api.banana.dev')

        self.assertEqual(find_param(proto, 'model_key'), 'model-key-1')

        self.assertIsNotNone(find_data_sample(proto, 'model_inputs'))
        self.assertIsNotNone(find_data_sample(proto, 'model_outputs'))
