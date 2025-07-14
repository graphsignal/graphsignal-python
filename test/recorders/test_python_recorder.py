import unittest
import logging
import sys
import os
import json
import time
from unittest.mock import patch, Mock
import pprint
import re

import graphsignal
from graphsignal.tracer import Tracer
from graphsignal.uploader import Uploader
from graphsignal.spans import Span
from graphsignal.recorders.python_recorder import PythonRecorder, _format_frame

logger = logging.getLogger('graphsignal')


class PythonRecorderTest(unittest.TestCase):
    def setUp(self):
        if len(logger.handlers) == 0:
            logger.addHandler(logging.StreamHandler(sys.stdout))
        graphsignal.configure(
            api_key='k1',
            profiling_rate=1,
            debug_mode=True)
        graphsignal._tracer.export_on_shutdown = False

    def tearDown(self):
        graphsignal.shutdown()

    @patch.object(Tracer, 'emit_span_start')
    @patch.object(Tracer, 'emit_span_stop')
    @patch.object(Tracer, 'emit_span_read')
    def test_record(self, mocked_emit_span_read, mocked_emit_span_stop, mocked_emit_span_start):
        recorder = PythonRecorder()
        recorder._exclude_path = 'donotmatchpath'
        recorder.setup()
        span = Span('op1')
        context = {}

        def slow_method():
            time.sleep(0.1)

        recorder.on_span_start(span, context)
        slow_method()
        slow_method()
        recorder.on_span_stop(span, context)
        recorder.on_span_read(span, context)

        slow_call = next(call for call in json.loads(span._profiles['profile.cpython'].content) if 'slow_method' in call['func_name'])
        self.assertTrue(slow_call['wall_time_ns'] > 0)
        self.assertTrue(slow_call['self_wall_time_ns'] > 0)

    def test_format_frame(self):
        self.assertEqual(_format_frame('p', 1, 'f'), 'f (p:1)')
        self.assertEqual(_format_frame('p', None, 'f'), 'f (p)')
        self.assertEqual(_format_frame(None, None, 'f'), 'f')
        self.assertEqual(_format_frame(None, None, None), 'unknown')