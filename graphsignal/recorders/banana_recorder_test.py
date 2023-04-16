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
from graphsignal.traces import DEFAULT_OPTIONS
from graphsignal.recorders.banana_recorder import BananaRecorder

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
        proto = signals_pb2.Trace()
        context = {}
        recorder.on_trace_start(proto, context, DEFAULT_OPTIONS)
        recorder.on_trace_stop(proto, context, DEFAULT_OPTIONS)
        recorder.on_trace_read(proto, context, DEFAULT_OPTIONS)

        self.assertEqual(proto.frameworks[0].name, 'Banana Python SDK')

    @patch.object(Uploader, 'upload_trace')
    @patch.object(banana, 'run')
    async def test_trace_run(self, mocked_run, mocked_upload_trace):
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

        proto = mocked_upload_trace.call_args[0][0]

        #pp = pprint.PrettyPrinter()
        #pp.pprint(MessageToJson(proto))

        self.assertEqual(proto.frameworks[0].name, 'Banana Python SDK')

        self.assertEqual(find_param(proto, 'model_key'), 'model-key-1')

        self.assertEqual(find_data_count(proto, 'model_inputs', 'byte_count'), 24.0)
        self.assertEqual(find_data_count(proto, 'model_inputs', 'element_count'), 2.0)
        self.assertEqual(find_data_count(proto, 'model_outputs', 'byte_count'), 65.0)
        self.assertEqual(find_data_count(proto, 'model_outputs', 'element_count'), 2.0)


def find_param(proto, name):
    for param in proto.params:
        if param.name == name:
            return param.value
    return None


def find_data_count(proto, data_name, count_name):
    for data_stats in proto.data_profile:
        if data_stats.data_name == data_name:
            for data_count in data_stats.counts:
                if data_count.name == count_name:
                    return data_count.count
    return None