import unittest
import logging
import sys
import os
import json
from unittest.mock import patch, Mock
import tensorflow as tf
from google.protobuf.json_format import MessageToJson
import pprint

import graphsignal
from graphsignal.profilers.tensorflow import profile_step
from graphsignal.proto import profiles_pb2
from graphsignal.uploader import Uploader

logger = logging.getLogger('graphsignal')


class TensorflowProfilerTest(unittest.TestCase):
    def setUp(self):
        if len(logger.handlers) == 0:
            logger.addHandler(logging.StreamHandler(sys.stdout))
        graphsignal.configure(
            api_key='k1',
            workload_name='w1',
            debug_mode=True)

    def tearDown(self):
        graphsignal.shutdown()

    @patch.object(Uploader, 'upload_profile')
    def test_profile_step(self, mocked_upload_profile):
        os.environ["TF_CONFIG"] = json.dumps({
            "cluster": {
                "chief": ["host1:port"],
                "worker": ["host1:port", "host2:port"]
            },
            "task": {"type": "worker", "index": 1}
        })

        @tf.function
        def f(x):
            while tf.reduce_sum(x) > 1:
                #tf.print(x)
                x = tf.tanh(x)
            return x

        with profile_step(phase_name='training', effective_batch_size=128, ensure_profile=True):
            f(tf.random.uniform([5]))

        profile = mocked_upload_profile.call_args[0][0]

        #pp = pprint.PrettyPrinter()
        #pp.pprint(MessageToJson(profile))

        self.assertEqual(
            profile.process_usage.ml_framework,
            profiles_pb2.ProcessUsage.MLFramework.TENSORFLOW)
        self.assertEqual(profile.process_usage.global_rank, 1)

        self.assertEqual(profile.phase_name, 'training')
        self.assertEqual(profile.step_stats.step_count, 1)
        self.assertTrue(profile.step_stats.total_time_us > 0)
        self.assertEqual(profile.step_stats.sample_count, 128)
        self.assertEqual(profile.step_stats.world_size, 3)

        test_op_stats = None
        for op_stats in profile.op_stats:
            if op_stats.op_name == 'RandomUniform':
                test_op_stats = op_stats
                break
        self.assertIsNotNone(test_op_stats)
        self.assertTrue(test_op_stats.count >= 1)
        if len(tf.config.list_physical_devices('GPU')) > 0:
            self.assertTrue(test_op_stats.total_device_time_us >= 1)
            self.assertTrue(test_op_stats.self_device_time_us >= 1)
        else:
            self.assertTrue(test_op_stats.total_host_time_us >= 1)
            self.assertTrue(test_op_stats.self_host_time_us >= 1)

        self.assertNotEqual(profile.trace_data, '')
