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
import asyncio

import graphsignal
from graphsignal.proto import signals_pb2
from graphsignal.uploader import Uploader
from graphsignal.spans import DEFAULT_OPTIONS
from graphsignal.recorders.yappi_recorder import YappiRecorder, _format_frame

logger = logging.getLogger('graphsignal')


class OpenAIRecorderTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        if len(logger.handlers) == 0:
            logger.addHandler(logging.StreamHandler(sys.stdout))
        graphsignal.configure(
            api_key='k1',
            deployment='d1',
            upload_on_shutdown=False,
            debug_mode=True)

    async def asyncTearDown(self):
        graphsignal.shutdown()

    async def test_record(self):
        recorder = YappiRecorder()
        recorder._exclude_path = 'donotmatchpath'
        recorder.setup()
        proto = signals_pb2.Span()
        context = {}

        async def slow_method():
            time.sleep(0.1)
            await asyncio.sleep(0.1)

        recorder.on_span_start(proto, context, graphsignal.TraceOptions(enable_profiling=True))
        await slow_method()
        await slow_method()
        recorder.on_span_stop(proto, context, graphsignal.TraceOptions(enable_profiling=True))
        recorder.on_span_read(proto, context, graphsignal.TraceOptions(enable_profiling=True))

        self.assertTrue('profiled' in proto.labels)

        slow_call = next(call for call in proto.op_profile if 'slow_method' in call.op_name)
        self.assertEqual(slow_call.op_type, signals_pb2.OpStats.OpType.PYTHON_OP)
        self.assertTrue(slow_call.self_host_time_ns > 0)
        self.assertTrue(slow_call.host_time_ns > 0)
        self.assertTrue(slow_call.self_host_time_percent > 0)

    async def test_format_frame(self):
        self.assertEqual(_format_frame('p', 1, 'f'), 'f (p:1)')
        self.assertEqual(_format_frame('p', None, 'f'), 'f (p)')
        self.assertEqual(_format_frame(None, None, 'f'), 'f')
        self.assertEqual(_format_frame(None, None, None), 'unknown')