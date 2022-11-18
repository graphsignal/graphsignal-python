import unittest
import logging
import sys
import os
import json
import time
from unittest.mock import patch, Mock
from google.protobuf.json_format import MessageToJson
import pprint

import graphsignal
from graphsignal.proto import signals_pb2
from graphsignal.uploader import Uploader
from graphsignal.recorders.onnxruntime_recorder import ONNXRuntimeRecorder

logger = logging.getLogger('graphsignal')


class ONNXRuntimeRecorderTest(unittest.TestCase):
    def setUp(self):
        if len(logger.handlers) == 0:
            logger.addHandler(logging.StreamHandler(sys.stdout))
        graphsignal.configure(
            api_key='k1',
            debug_mode=True)

    def tearDown(self):
        graphsignal.shutdown()

    def test_record(self):
        recorder = ONNXRuntimeRecorder()
        recorder.setup()
        signal = signals_pb2.WorkerSignal()
        context = {}
        recorder.on_trace_start(signal, context)
        recorder.on_trace_stop(signal, context)

        self.assertEqual(
            signal.frameworks[0].type,
            signals_pb2.FrameworkInfo.FrameworkType.ONNX_FRAMEWORK)
