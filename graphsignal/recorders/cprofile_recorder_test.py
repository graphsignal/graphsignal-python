import unittest
import logging
import sys
import os
import json
import time
from unittest.mock import patch, Mock
from google.protobuf.json_format import MessageToJson
import pprint
import re

import graphsignal
from graphsignal.proto import signals_pb2
from graphsignal.uploader import Uploader
from graphsignal.recorders.cprofile_recorder import CProfileRecorder, _format_frame

logger = logging.getLogger('graphsignal')


class CProfileRecorderTest(unittest.TestCase):
    def setUp(self):
        if len(logger.handlers) == 0:
            logger.addHandler(logging.StreamHandler(sys.stdout))
        graphsignal.configure(
            api_key='k1',
            debug_mode=True)

    def tearDown(self):
        graphsignal.shutdown()

    def test_record(self):
        recorder = CProfileRecorder()
        recorder._exclude_path = 'donotmatchpath'
        recorder.setup()
        signal = signals_pb2.WorkerSignal()
        context = {}

        def slow_method():
            time.sleep(0.1)

        recorder.on_trace_start(signal, context)
        slow_method()
        slow_method()
        recorder.on_trace_stop(signal, context)
        recorder.on_trace_read(signal, context)

        #pp = pprint.PrettyPrinter()
        #pp.pprint(MessageToJson(signal))

        slow_op = next(op for op in signal.operations if 'slow_method' in op.op_name)
        self.assertTrue(slow_op.total_self_wall_time_ns > 0)
        self.assertTrue(slow_op.total_cum_wall_time_ns > 0)
        self.assertTrue(slow_op.total_self_wall_time_percent > 0)


    def test_format_frame(self):
        self.assertEqual(_format_frame('p', 1, 'f'), 'f (p:1)')
        self.assertEqual(_format_frame('p', None, 'f'), 'f (p)')
        self.assertEqual(_format_frame(None, None, 'f'), 'f')
        self.assertEqual(_format_frame(None, None, None), 'unknown')