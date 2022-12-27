import unittest
import logging
import sys
from unittest.mock import patch, Mock
import torch
from google.protobuf.json_format import MessageToJson
import pprint
import time
import random

import graphsignal
from graphsignal.proto import signals_pb2
from graphsignal.endpoint_trace import DEFAULT_OPTIONS
from graphsignal.recorders.process_recorder import ProcessRecorder

logger = logging.getLogger('graphsignal')

mem = []

class ProcessRecorderTest(unittest.TestCase):
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
        recorder = ProcessRecorder()
        recorder.setup()
        signal = signals_pb2.WorkerSignal()
        context = {}
        recorder.on_trace_start(signal, context, DEFAULT_OPTIONS)
        random.random()
        recorder.on_trace_stop(signal, context, DEFAULT_OPTIONS)
        recorder.on_trace_read(signal, context, DEFAULT_OPTIONS)

        ProcessRecorder.MIN_CPU_READ_INTERVAL_US = 0
        time.sleep(0.2)

        signal = signals_pb2.WorkerSignal()
        context = {}
        recorder.on_trace_start(signal, context, DEFAULT_OPTIONS)
        for _ in range(100000):
            random.random()
        global mem
        mem = [1]*100000
        recorder.on_trace_stop(signal, context, DEFAULT_OPTIONS)
        recorder.on_trace_read(signal, context, DEFAULT_OPTIONS)

        #pp = pprint.PrettyPrinter()
        #pp.pprint(MessageToJson(signal))

        self.assertNotEqual(signal.node_usage.hostname, '')
        self.assertNotEqual(signal.node_usage.ip_address, '')
        self.assertNotEqual(signal.process_usage.pid, '')
        if sys.platform != 'win32':
            self.assertTrue(signal.trace_sample.thread_cpu_time_us > 0)
            self.assertTrue(signal.node_usage.mem_total > 0)
            self.assertTrue(signal.node_usage.mem_used > 0)
            self.assertTrue(signal.process_usage.cpu_usage_percent > 0)
            self.assertTrue(signal.process_usage.current_rss > 0)
            self.assertTrue(signal.process_usage.max_rss > 0)
            self.assertTrue(signal.process_usage.vm_size > 0)
