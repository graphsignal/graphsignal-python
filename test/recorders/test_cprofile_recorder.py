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
from graphsignal.spans import DEFAULT_OPTIONS
from graphsignal.recorders.cprofile_recorder import CProfileRecorder, _format_frame

logger = logging.getLogger('graphsignal')


class CProfileRecorderTest(unittest.TestCase):
    def setUp(self):
        if len(logger.handlers) == 0:
            logger.addHandler(logging.StreamHandler(sys.stdout))
        graphsignal.configure(
            api_key='k1',
            deployment='d1',
            upload_on_shutdown=False,
            debug_mode=True)

    def tearDown(self):
        graphsignal.shutdown()

    def test_record(self):
        recorder = CProfileRecorder()
        recorder._exclude_path = 'donotmatchpath'
        recorder.setup()
        proto = signals_pb2.Span()
        context = {}

        def slow_method():
            time.sleep(0.1)

        recorder.on_span_start(proto, context, graphsignal.TraceOptions(enable_profiling=True))
        slow_method()
        slow_method()
        recorder.on_span_stop(proto, context, graphsignal.TraceOptions(enable_profiling=True))
        recorder.on_span_read(proto, context, graphsignal.TraceOptions(enable_profiling=True))

        self.assertTrue('profiled' in proto.labels)

        slow_call = next(call for call in proto.op_profile if 'slow_method' in call.op_name)
        self.assertEqual(slow_call.op_type, signals_pb2.OpStats.OpType.PYTHON_OP)
        self.assertTrue(slow_call.self_host_time_ns > 0)
        self.assertTrue(slow_call.host_time_ns > 0)
        self.assertTrue(slow_call.self_host_time_percent > 0)

    def test_format_frame(self):
        self.assertEqual(_format_frame('p', 1, 'f'), 'f (p:1)')
        self.assertEqual(_format_frame('p', None, 'f'), 'f (p)')
        self.assertEqual(_format_frame(None, None, 'f'), 'f')
        self.assertEqual(_format_frame(None, None, None), 'unknown')