import unittest
import logging
import sys
from unittest.mock import patch, Mock
import torch
from google.protobuf.json_format import MessageToJson
import pprint

import graphsignal
from graphsignal.profilers.pytorch import PyTorchProfiler
from graphsignal.proto import signals_pb2
from graphsignal.uploader import Uploader

logger = logging.getLogger('graphsignal')


class PyTorchProfilerTest(unittest.TestCase):
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

    def test_read_info(self):
        profiler = PyTorchProfiler()
        signal = signals_pb2.WorkerSignal()
        profiler.read_info(signal)

        self.assertEqual(
            signal.frameworks[0].type,
            signals_pb2.FrameworkInfo.FrameworkType.PYTORCH_FRAMEWORK)

    def test_start_stop(self):
        x = torch.arange(-5, 5, 0.1).view(-1, 1)
        y = -5 * x + 0.1 * torch.randn(x.size())
        model = torch.nn.Linear(1, 1)
        if torch.cuda.is_available():
            x = x.to('cuda:0')
            y = y.to('cuda:0')
            model = model.to('cuda:0')

        profiler = PyTorchProfiler()
        signal = signals_pb2.WorkerSignal()
        profiler.start(signal)
        y1 = model(x)
        profiler.stop(signal)

        #pp = pprint.PrettyPrinter()
        #pp.pprint(MessageToJson(signal))

        self.assertTrue(sum([op_stats.count for op_stats in signal.op_stats]) > 0)
        if torch.cuda.is_available():
            self.assertTrue(sum([op_stats.total_device_time_us for op_stats in signal.op_stats]) > 0)
            self.assertTrue(sum([op_stats.self_device_time_us for op_stats in signal.op_stats]) > 0)
        else:
            self.assertTrue(sum([op_stats.total_host_time_us for op_stats in signal.op_stats]) > 0)
            self.assertTrue(sum([op_stats.self_host_time_us for op_stats in signal.op_stats]) > 0)
        self.assertNotEqual(signal.trace_data, b'')
