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
from graphsignal.profilers.tensorflow import TensorFlowProfiler
from graphsignal.proto import signals_pb2
from graphsignal.uploader import Uploader

logger = logging.getLogger('graphsignal')


class TensorFlowProfilerTest(unittest.TestCase):
    def setUp(self):
        if len(logger.handlers) == 0:
            logger.addHandler(logging.StreamHandler(sys.stdout))
        graphsignal.configure(
            api_key='k1',
            debug_mode=True)

    def tearDown(self):
        graphsignal.shutdown()

    def test_read_info(self):
        os.environ["TF_CONFIG"] = json.dumps({
            "cluster": {
                "chief": ["host1:port"],
                "worker": ["host1:port", "host2:port"]
            },
            "task": {"type": "worker", "index": 1}
        })

        profiler = TensorFlowProfiler()
        signal = signals_pb2.MLSignal()
        profiler.read_info(signal)

        self.assertEqual(
            signal.frameworks[0].type,
            signals_pb2.FrameworkInfo.FrameworkType.TENSORFLOW_FRAMEWORK)
        self.assertEqual(signal.process_usage.global_rank, 1)

        self.assertEqual(signal.cluster_info.world_size, 3)

    def test_start_stop(self):
        @tf.function
        def f(x):
            while tf.reduce_sum(x) > 1:
                #tf.print(x)
                x = tf.tanh(x)
            return x

        profiler = TensorFlowProfiler()
        signal = signals_pb2.MLSignal()
        profiler.start(signal)
        f(tf.random.uniform([5]))
        profiler.stop(signal)

        #pp = pprint.PrettyPrinter()
        #pp.pprint(MessageToJson(signal))

        test_op_stats = None
        for op_stats in signal.op_stats:
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

        self.assertNotEqual(signal.trace_data, b'')
