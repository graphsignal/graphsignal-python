import unittest
import logging
import sys
from unittest.mock import patch, Mock
import tensorflow as tf
from google.protobuf.json_format import MessageToJson
import pprint

import graphsignal
from graphsignal.profilers.tensorflow_profiler import TensorflowProfiler
from graphsignal.proto import profiles_pb2

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

    def test_start_stop(self):
        profiler = TensorflowProfiler()

        @tf.function
        def f(x):
            while tf.reduce_sum(x) > 1:
                tf.print(x)
                x = tf.tanh(x)
            return x

        profiler.start()
        f(tf.random.uniform([5]))
        profile = profiles_pb2.MLProfile()
        profiler.stop(profile)

        #pp = pprint.PrettyPrinter()
        # pp.pprint(MessageToJson(profile))

        self.assertEqual(
            profile.run_env.ml_framework,
            profiles_pb2.RunEnvironment.MLFramework.TENSORFLOW)
        if len(tf.config.list_physical_devices('GPU')) > 0:
            self.assertEqual(
                profile.run_env.devices[0].type,
                profiles_pb2.DeviceType.GPU)
        else:
            self.assertEqual(len(profile.run_env.devices), 0)

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
