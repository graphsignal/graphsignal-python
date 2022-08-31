import unittest
import logging
import sys
import os
from unittest.mock import patch, Mock
import torch
import onnxruntime
from google.protobuf.json_format import MessageToJson
import pprint

import graphsignal
from graphsignal.profilers.onnxruntime import ONNXRuntimeProfiler
from graphsignal.proto import signals_pb2
from graphsignal.uploader import Uploader

logger = logging.getLogger('graphsignal')

TEST_MODEL_PATH = '/tmp/test_model.onnx'

class ONNXRuntimeProfilerTest(unittest.TestCase):
    def setUp(self):
        if len(logger.handlers) == 0:
            logger.addHandler(logging.StreamHandler(sys.stdout))
        graphsignal.configure(
            api_key='k1',
            debug_mode=True)

    def tearDown(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        graphsignal.shutdown()
        if os.path.exists(TEST_MODEL_PATH):
            os.remove(TEST_MODEL_PATH)

    def test_read_info(self):
        profiler = ONNXRuntimeProfiler()
        signal = signals_pb2.MLSignal()
        profiler.read_info(signal)

        self.assertEqual(
            signal.frameworks[0].type,
            signals_pb2.FrameworkInfo.FrameworkType.ONNX_FRAMEWORK)

    def test_start_stop(self):
        x = torch.arange(-5, 5, 0.1).view(-1, 1)
        y = -5 * x + 0.1 * torch.randn(x.size())
        model = torch.nn.Linear(1, 1)
        if torch.cuda.is_available():
            x = x.to('cuda:0')
            y = y.to('cuda:0')
            model = model.to('cuda:0')
        y1 = model(x)

        input_names = [ "input" ]
        output_names = [ "output" ]
        torch.onnx.export(model, x, TEST_MODEL_PATH, verbose=True, input_names=input_names, output_names=output_names)

        profiler = ONNXRuntimeProfiler()
        signal = signals_pb2.MLSignal()

        sess_options = onnxruntime.SessionOptions()
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

        profiler.initialize_options(sess_options)
        session = onnxruntime.InferenceSession(TEST_MODEL_PATH, sess_options)

        profiler.set_onnx_session(session)

        profiler.start(signal)
        session.run(None, { 'input': x.detach().cpu().numpy() })
        profiler.stop(signal)

        #pp = pprint.PrettyPrinter()
        #pp.pprint(MessageToJson(signal))

        self.assertNotEqual(signal.trace_data, b'')
