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
from graphsignal.profilers.onnxruntime import initialize_profiler, profile_inference
from graphsignal.proto import profiles_pb2
from graphsignal.uploader import Uploader

logger = logging.getLogger('graphsignal')

TEST_MODEL_PATH = '/tmp/test_model.onnx'

class ONNXRuntimeProfilerTest(unittest.TestCase):
    def setUp(self):
        if len(logger.handlers) == 0:
            logger.addHandler(logging.StreamHandler(sys.stdout))
        graphsignal.configure(
            api_key='k1',
            workload_name='w1',
            debug_mode=True)

    def tearDown(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        graphsignal.shutdown()
        os.remove(TEST_MODEL_PATH)

    @patch.object(Uploader, 'upload_profile')
    def test_profile_inference(self, mocked_upload_profile):
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

        with profile_inference(session):
            session.run(None, { 'input': x.detach().cpu().numpy() })

        with profile_inference(session, ensure_profile=True):
            session.run(None, { 'input': x.detach().cpu().numpy() })

        graphsignal.upload()
        profile = mocked_upload_profile.call_args[0][0]

        #pp = pprint.PrettyPrinter()
        #pp.pprint(MessageToJson(profile))

        self.assertEqual(
            profile.frameworks[0].type,
            profiles_pb2.FrameworkInfo.FrameworkType.ONNX_FRAMEWORK)

        self.assertNotEqual(profile.trace_data, b'')
