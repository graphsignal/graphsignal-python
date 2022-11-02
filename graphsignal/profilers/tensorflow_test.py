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
        profiler = TensorFlowProfiler()
        signal = signals_pb2.WorkerSignal()
        profiler.read_info(signal)

        self.assertEqual(
            signal.frameworks[0].type,
            signals_pb2.FrameworkInfo.FrameworkType.TENSORFLOW_FRAMEWORK)

    def test_start_stop(self):
        @tf.function
        def f(x):
            while tf.reduce_sum(x) > 1:
                #tf.print(x)
                x = tf.tanh(x)
            return x

        profiler = TensorFlowProfiler()
        signal = signals_pb2.WorkerSignal()
        profiler.start(signal)
        f(tf.random.uniform([5]))
        profiler.stop(signal)

        #pp = pprint.PrettyPrinter()
        #pp.pprint(MessageToJson(signal))

        self.assertTrue(sum([op_stats.count for op_stats in signal.op_stats]) > 0)
        if len(tf.config.list_physical_devices('GPU')) > 0:
            self.assertTrue(sum([op_stats.total_device_time_us for op_stats in signal.op_stats]) > 0)
            self.assertTrue(sum([op_stats.self_device_time_us for op_stats in signal.op_stats]) > 0)
        else:
            self.assertTrue(sum([op_stats.total_host_time_us for op_stats in signal.op_stats]) > 0)
            self.assertTrue(sum([op_stats.self_host_time_us for op_stats in signal.op_stats]) > 0)
        self.assertNotEqual(signal.trace_data, b'')
