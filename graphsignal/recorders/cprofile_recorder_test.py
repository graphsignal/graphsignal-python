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
from graphsignal.endpoint_trace import DEFAULT_OPTIONS
from graphsignal.recorders.cprofile_recorder import CProfileRecorder

logger = logging.getLogger('graphsignal')


class CProfileRecorderTest(unittest.TestCase):
    def setUp(self):
        if len(logger.handlers) == 0:
            logger.addHandler(logging.StreamHandler(sys.stdout))
        graphsignal.configure(
            api_key='k1',
            deployment='d1',
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

        recorder.on_trace_start(signal, context, graphsignal.TraceOptions(enable_profiling=True))
        slow_method()
        slow_method()
        recorder.on_trace_stop(signal, context, graphsignal.TraceOptions(enable_profiling=True))
        recorder.on_trace_read(signal, context, graphsignal.TraceOptions(enable_profiling=True))

        #pp = pprint.PrettyPrinter()
        #pp.pprint(MessageToJson(signal))

        self.assertEqual(signal.call_profile.profile_type, signals_pb2.Profile.PROFILE_TYPE_PYTHON)
        slow_call = next(call for call in signal.call_profile.stats if 'slow_method' in call.func_name)
        self.assertTrue(slow_call.total_self_wall_time_ns > 0)
        self.assertTrue(slow_call.total_cum_wall_time_ns > 0)
        self.assertTrue(slow_call.total_self_wall_time_percent > 0)
