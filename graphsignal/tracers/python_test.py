import unittest
import logging
import sys
import time
from unittest.mock import patch, Mock
from google.protobuf.json_format import MessageToJson
import pprint

import graphsignal
from graphsignal.tracers.python import inference_span
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

    @patch.object(Uploader, 'upload_signal')
    def test_inference_span(self, mocked_upload_signal):
        def slow_method():
            time.sleep(0.1)

        graphsignal.tracers.python._profiler._exclude_path = 'donotmatchpath'
        with inference_span('m1'):
            slow_method()
            slow_method()

        signal = mocked_upload_signal.call_args[0][0]

        #pp = pprint.PrettyPrinter()
        #pp.pprint(MessageToJson(signal))

        foundOp = False
        for op_stats in signal.op_stats:
            if op_stats.op_name.startswith('slow_method') and op_stats.count == 2 and op_stats.total_host_time_us > 200000:
                foundOp = True
                break
        self.assertTrue(foundOp)
