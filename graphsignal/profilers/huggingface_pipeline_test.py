import unittest
import logging
import sys
from unittest.mock import patch, Mock
from google.protobuf.json_format import MessageToJson
import pprint

import graphsignal
from graphsignal.proto import signals_pb2
from graphsignal.uploader import Uploader

logger = logging.getLogger('graphsignal')


class HuggingFaceGeneratorTest(unittest.TestCase):
    def setUp(self):
        if len(logger.handlers) == 0:
            logger.addHandler(logging.StreamHandler(sys.stdout))
        graphsignal.configure(
            api_key='k1',
            debug_mode=True)

    def tearDown(self):
        graphsignal.shutdown()

    def test_pipeline(self):
        from transformers import pipeline
        from graphsignal.profilers.pytorch import PyTorchProfiler

        pipe = pipeline(task="text-generation", model='distilgpt2')

        profiler = PyTorchProfiler()
        signal = signals_pb2.WorkerSignal()
        profiler.start(signal)
        output = pipe('some text')
        profiler.stop(signal)

        #pp = pprint.PrettyPrinter()
        #pp.pprint(MessageToJson(signal))

        self.assertTrue(len(signal.op_stats) > 0)
