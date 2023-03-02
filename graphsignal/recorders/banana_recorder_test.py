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
from graphsignal.endpoint_trace import DEFAULT_OPTIONS
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
        signal = signals_pb2.WorkerSignal()
        context = {}
        recorder.on_trace_start(signal, context, DEFAULT_OPTIONS)
        recorder.on_trace_stop(signal, context, DEFAULT_OPTIONS)
        recorder.on_trace_read(signal, context, DEFAULT_OPTIONS)

        self.assertEqual(signal.frameworks[0].name, 'Banana Python SDK')

    @patch.object(Uploader, 'upload_signal')
    @patch.object(banana, 'run')
    async def test_trace_run(self, mocked_run, mocked_upload_signal):
        # mocking overrides autoinstrumentation, reinstrument
        recorder = BananaRecorder()
        recorder.setup()
        recorder._is_sampling = True

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

        signal = mocked_upload_signal.call_args[0][0]

        #pp = pprint.PrettyPrinter()
        #pp.pprint(MessageToJson(signal))

        self.assertEqual(signal.frameworks[0].name, 'Banana Python SDK')

        self.assertEqual(find_param(signal, 'model_key'), 'model-key-1')

        self.assertEqual(find_data_metric(signal, 'model_inputs', 'byte_count'), 24.0)
        self.assertEqual(find_data_metric(signal, 'model_inputs', 'element_count'), 2.0)
        self.assertEqual(find_data_metric(signal, 'model_outputs', 'byte_count'), 65.0)
        self.assertEqual(find_data_metric(signal, 'model_outputs', 'element_count'), 2.0)


def find_param(signal, name):
    for param in signal.params:
        if param.name == name:
            return param.value
    return None


def find_data_metric(signal, data_name, metric_name):
    for data_metric in signal.data_metrics:
        if data_metric.data_name == data_name and data_metric.metric_name == metric_name:
            return data_metric.metric.gauge
    return None