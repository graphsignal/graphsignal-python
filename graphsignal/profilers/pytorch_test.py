import unittest
import logging
import sys
from unittest.mock import patch, Mock
import torch
from google.protobuf.json_format import MessageToJson
import pprint

import graphsignal
from graphsignal.profilers.pytorch import profile_step
from graphsignal.proto import profiles_pb2
from graphsignal.uploader import Uploader

logger = logging.getLogger('graphsignal')


class PyTorchProfilerTest(unittest.TestCase):
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

    @patch.object(Uploader, 'upload_profile')
    def test_profile_step(self, mocked_upload_profile):
        x = torch.arange(-5, 5, 0.1).view(-1, 1)
        y = -5 * x + 0.1 * torch.randn(x.size())
        model = torch.nn.Linear(1, 1)
        if torch.cuda.is_available():
            x = x.to('cuda:0')
            y = y.to('cuda:0')
            model = model.to('cuda:0')
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        with profile_step('training' , effective_batch_size=128, ensure_profile=True):
            y1 = model(x)
            loss = criterion(y1, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        profile = mocked_upload_profile.call_args[0][0]

        #pp = pprint.PrettyPrinter()
        #pp.pprint(MessageToJson(profile))

        self.assertEqual(
            profile.process_usage.ml_framework,
            profiles_pb2.ProcessUsage.MLFramework.PYTORCH)

        self.assertEqual(profile.phase_name, 'training')
        self.assertEqual(profile.step_stats.step_count, 1)
        self.assertTrue(profile.step_stats.total_time_us > 0)
        self.assertEqual(profile.step_stats.sample_count, 128)

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
        else:
            self.assertTrue(test_op_stats.total_host_time_us >= 1)
            self.assertTrue(test_op_stats.self_host_time_us >= 1)

        self.assertNotEqual(profile.trace_data, '')
