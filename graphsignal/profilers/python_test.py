import unittest
import logging
import sys
import time
from unittest.mock import patch, Mock
from google.protobuf.json_format import MessageToJson
import pprint

import graphsignal
from graphsignal.profilers.python import PythonProfiler
from graphsignal.proto import signals_pb2
from graphsignal.uploader import Uploader

logger = logging.getLogger('graphsignal')


class PythonProfilerTest(unittest.TestCase):
    def setUp(self):
        if len(logger.handlers) == 0:
            logger.addHandler(logging.StreamHandler(sys.stdout))
        graphsignal.configure(
            api_key='k1',
            debug_mode=True)

    def tearDown(self):
        graphsignal.shutdown()

    def test_stat_stop(self):
        def slow_method():
            time.sleep(0.1)

        profiler = PythonProfiler()
        profiler._exclude_path = 'donotmatchpath'
        signal = signals_pb2.MLSignal()
        profiler.start(signal)
        slow_method()
        slow_method()
        profiler.stop(signal)

        #pp = pprint.PrettyPrinter()
        #pp.pprint(MessageToJson(signal))

        foundOp = False
        for op_stats in signal.op_stats:
            if op_stats.op_name.startswith('slow_method') and op_stats.count == 2 and op_stats.total_host_time_us > 200000:
                foundOp = True
                break
        self.assertTrue(foundOp)
