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

        test_op_stats = None
        for op_stats in signal.op_stats:
            if op_stats.op_name == 'aten::addmm':
                test_op_stats = op_stats
                break
        self.assertIsNotNone(test_op_stats)
        self.assertTrue(test_op_stats.count >= 1)
        if torch.cuda.is_available():
            self.assertTrue(test_op_stats.total_device_time_us >= 1)
            self.assertTrue(test_op_stats.self_device_time_us >= 1)
        else:
            self.assertTrue(test_op_stats.total_host_time_us >= 1)
            self.assertTrue(test_op_stats.self_host_time_us >= 1)

        self.assertNotEqual(signal.trace_data, b'')
