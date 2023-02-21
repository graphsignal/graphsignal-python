import unittest
import logging
import sys
import os
import json
import time
from unittest.mock import patch, Mock
from google.protobuf.json_format import MessageToJson
import pprint
import tensorflow as tf

import graphsignal
from graphsignal.proto import signals_pb2
from graphsignal.uploader import Uploader
from graphsignal.endpoint_trace import DEFAULT_OPTIONS
from graphsignal.recorders.tensorflow_recorder import TensorFlowRecorder

logger = logging.getLogger('graphsignal')


class TensorFlowRecorderTest(unittest.TestCase):
    def setUp(self):
        if len(logger.handlers) == 0:
            logger.addHandler(logging.StreamHandler(sys.stdout))
        graphsignal.configure(
            api_key='k1',
            deployment='d1',
            debug_mode=True)

    def tearDown(self):
        graphsignal.shutdown()

    def test_record(self):
        os.environ["TF_CONFIG"] = json.dumps({
            "cluster": {
                "chief": ["host1:port"],
                "worker": ["host1:port", "host2:port"]
            },
            "task": {"type": "worker", "index": 1}
        })

        recorder = TensorFlowRecorder()
        recorder.setup()
        signal = signals_pb2.WorkerSignal()
        context = {}
        recorder.on_trace_start(signal, context, DEFAULT_OPTIONS)
        recorder.on_trace_stop(signal, context, DEFAULT_OPTIONS)
        recorder.on_trace_read(signal, context, DEFAULT_OPTIONS)

        self.assertEqual(signal.frameworks[0].name, 'TensorFlow')

        self.assertEqual(signal.frameworks[0].params[0].name, 'cluster_size')
        self.assertEqual(signal.frameworks[0].params[0].value, '3')
        self.assertEqual(signal.frameworks[0].params[1].name, 'task_index')
        self.assertEqual(signal.frameworks[0].params[1].value, '1')
        self.assertEqual(signal.frameworks[0].params[2].name, 'tf.test.is_built_with_gpu_support')
        self.assertEqual(signal.frameworks[0].params[2].value, str(tf.test.is_built_with_gpu_support()))
