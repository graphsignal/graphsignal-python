import unittest
import logging
import sys
import os
import json
import time
from unittest.mock import patch, Mock
from google.protobuf.json_format import MessageToJson
import pprint
import torch

import graphsignal
from graphsignal.proto import signals_pb2
from graphsignal.uploader import Uploader
from graphsignal.spans import DEFAULT_OPTIONS
from graphsignal.recorders.kineto_recorder import KinetoRecorder

logger = logging.getLogger('graphsignal')


class KinetoRecorderTest(unittest.TestCase):
    def setUp(self):
        if len(logger.handlers) == 0:
            logger.addHandler(logging.StreamHandler(sys.stdout))
        graphsignal.configure(
            api_key='k1',
            deployment='d1',
            upload_on_shutdown=False,
            debug_mode=True)

    def tearDown(self):
        graphsignal.shutdown()

    def test_record(self):
        recorder = KinetoRecorder()
        recorder.setup()

        x = torch.arange(-5, 5, 0.1).view(1, 1, -1, 1)
        model = torch.nn.Conv2d(1, 1, kernel_size=(1, 1))
        if torch.cuda.is_available():
            x = x.to('cuda:0')
            model = model.to('cuda:0')

        proto = signals_pb2.Span()
        context = {}
        recorder.on_span_start(proto, context, graphsignal.TraceOptions(enable_profiling=True))

        pred = model(x)

        recorder.on_span_stop(proto, context, graphsignal.TraceOptions(enable_profiling=True))
        recorder.on_span_read(proto, context, graphsignal.TraceOptions(enable_profiling=True))

        self.assertTrue('profiled' in proto.labels)

        test_op_stats = None
        for op_stats in proto.op_profile:
            print(op_stats.op_name)
            if op_stats.op_name == 'aten::conv2d':
                test_op_stats = op_stats
                break
        self.assertIsNotNone(test_op_stats)
        self.assertTrue(test_op_stats.count >= 1)
        
        self.assertEqual(test_op_stats.op_type, signals_pb2.OpStats.OpType.PYTORCH_OP)
        if torch.cuda.is_available():
            self.assertTrue(test_op_stats.device_time_ns >= 1)
            self.assertTrue(test_op_stats.self_device_time_ns >= 1)
        else:
            self.assertTrue(test_op_stats.host_time_ns >= 1)
            self.assertTrue(test_op_stats.self_host_time_ns >= 1)
            self.assertTrue(test_op_stats.self_host_time_percent > 0)
