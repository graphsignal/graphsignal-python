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
from graphsignal.spans import DEFAULT_OPTIONS
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
            upload_on_shutdown=False,
            debug_mode=True)

    def tearDown(self):
        graphsignal.shutdown()

    def test_record(self):
        recorder = ProcessRecorder()
        recorder.setup()

        ProcessRecorder.MIN_CPU_READ_INTERVAL_US = 0
        time.sleep(0.2)
        for _ in range(100000):
            random.random()
        global mem
        mem = [1]*100000

        recorder.take_snapshot()

        proto = signals_pb2.Span()
        context = {}
        recorder.on_span_start(proto, context, DEFAULT_OPTIONS)
        recorder.on_span_stop(proto, context, DEFAULT_OPTIONS)
        recorder.on_span_read(proto, context, DEFAULT_OPTIONS)

        self.assertNotEqual(proto.node_usage.hostname, '')
        self.assertNotEqual(proto.node_usage.ip_address, '')
        self.assertNotEqual(proto.process_usage.pid, '')
        if sys.platform != 'win32':
            self.assertTrue(proto.node_usage.mem_total > 0)
            self.assertTrue(proto.node_usage.mem_used > 0)
            self.assertTrue(proto.process_usage.cpu_usage_percent > 0)
            self.assertTrue(proto.process_usage.current_rss > 0)
            self.assertTrue(proto.process_usage.max_rss > 0)
            self.assertTrue(proto.process_usage.vm_size > 0)

        recorder.on_metric_update()

        store = graphsignal._tracer.metric_store()
        metric_tags =  {'deployment': 'd1', 'hostname': proto.node_usage.hostname}
        key = store.metric_key('system', 'process_cpu_usage', metric_tags)
        self.assertTrue(store._metrics[key].gauge > 0)
        key = store.metric_key('system', 'process_memory', metric_tags)
        self.assertTrue(store._metrics[key].gauge > 0)
        key = store.metric_key('system', 'virtual_memory', metric_tags)
        self.assertTrue(store._metrics[key].gauge > 0)
        key = store.metric_key('system', 'node_memory_used', metric_tags)
        self.assertTrue(store._metrics[key].gauge > 0)
