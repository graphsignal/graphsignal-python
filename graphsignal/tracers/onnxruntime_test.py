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
from graphsignal.tracers.onnxruntime import initialize_profiler, inference_span
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
        os.remove(TEST_MODEL_PATH)

    @patch.object(Uploader, 'upload_signal')
    def test_inference_span(self, mocked_upload_signal):
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

        sess_options = onnxruntime.SessionOptions()
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

        initialize_profiler(sess_options)

        session = onnxruntime.InferenceSession(TEST_MODEL_PATH, sess_options)

        with inference_span(model_name='m1', onnx_session=session):
            session.run(None, { 'input': x.detach().cpu().numpy() })

        signal = mocked_upload_signal.call_args[0][0]

        with inference_span(model_name='m1', ensure_trace=True, onnx_session=session):
            session.run(None, { 'input': x.detach().cpu().numpy() })

        #pp = pprint.PrettyPrinter()
        #pp.pprint(MessageToJson(signal))

        self.assertEqual(
            signal.frameworks[0].type,
            signals_pb2.FrameworkInfo.FrameworkType.ONNX_FRAMEWORK)

        self.assertNotEqual(signal.trace_data, b'')
