import unittest
import logging
import sys
import os
import json
import time
from unittest.mock import patch, Mock
from google.protobuf.json_format import MessageToJson
import pprint
import xgboost as xgb

import graphsignal
from graphsignal.proto import signals_pb2
from graphsignal.uploader import Uploader
from graphsignal.endpoint_trace import DEFAULT_OPTIONS
from graphsignal.recorders.xgboost_recorder import XGBoostRecorder

logger = logging.getLogger('graphsignal')


class XGBoostRecorderTest(unittest.TestCase):
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
        recorder = XGBoostRecorder()
        recorder.setup()
        signal = signals_pb2.Trace()
        context = {}
        recorder.on_trace_start(signal, context, DEFAULT_OPTIONS)
        recorder.on_trace_stop(signal, context, DEFAULT_OPTIONS)
        recorder.on_trace_read(signal, context, DEFAULT_OPTIONS)

        self.assertEqual(signal.frameworks[0].name, 'XGBoost')

        self.assertTrue(len(signal.frameworks[0].params) > 0)
