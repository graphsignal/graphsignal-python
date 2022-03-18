import unittest
import logging
import sys
from unittest.mock import patch, Mock
import torch
from google.protobuf.json_format import MessageToJson
import pprint

import graphsignal
from graphsignal.profilers.pytorch_profiler import PytorchProfiler
from graphsignal.proto import profiles_pb2

logger = logging.getLogger('graphsignal')


class PytorchProfilerTest(unittest.TestCase):
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

    def test_start_stop(self):
        profiler = PytorchProfiler()

        x = torch.arange(-5, 5, 0.1).view(-1, 1)
        y = -5 * x + 0.1 * torch.randn(x.size())
        model = torch.nn.Linear(1, 1)
        if torch.cuda.is_available():
            x = x.to('cuda:0')
            y = y.to('cuda:0')
            model = model.to('cuda:0')
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        profiler.start()

        y1 = model(x)
        loss = criterion(y1, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        profile = profiles_pb2.MLProfile()
        profiler.stop(profile)

        #pp = pprint.PrettyPrinter()
        #pp.pprint(MessageToJson(profile))

        self.assertEqual(
            profile.run_env.ml_framework,
            profiles_pb2.RunEnvironment.MLFramework.PYTORCH)
        if torch.cuda.is_available():
            self.assertEqual(
                profile.run_env.devices[0].type,
                profiles_pb2.DeviceType.GPU)
        else:
            self.assertEqual(len(profile.run_env.devices), 0)

        test_op_stats = None
        for op_stats in profile.op_stats:
            if op_stats.op_name == 'aten::mse_loss':
                test_op_stats = op_stats
                break
        self.assertIsNotNone(test_op_stats)
        self.assertTrue(test_op_stats.count >= 1)
        if torch.cuda.is_available():
            self.assertTrue(test_op_stats.total_device_time_us >= 1)
            self.assertTrue(test_op_stats.self_device_time_us >= 1)
            self.assertTrue(profile.summary.device_op_percent > 0)
        else:
            self.assertTrue(test_op_stats.total_host_time_us >= 1)
            self.assertTrue(test_op_stats.self_host_time_us >= 1)
            self.assertEqual(profile.summary.host_op_percent, 100)
