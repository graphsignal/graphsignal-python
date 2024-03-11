import unittest
import logging
import sys
from unittest.mock import patch, Mock
import socket
from google.protobuf.json_format import MessageToJson
import pprint
import time
import random

import graphsignal
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

        recorder.on_metric_update()

        store = graphsignal._tracer.metric_store()
        metric_tags =  {'deployment': 'd1', 'hostname': socket.gethostname()}
        key = store.metric_key('system', 'process_cpu_usage', metric_tags)
        self.assertTrue(store._metrics[key].gauge > 0)
        key = store.metric_key('system', 'process_memory', metric_tags)
        self.assertTrue(store._metrics[key].gauge > 0)
        key = store.metric_key('system', 'virtual_memory', metric_tags)
        self.assertTrue(store._metrics[key].gauge > 0)
        key = store.metric_key('system', 'node_memory_used', metric_tags)
        self.assertTrue(store._metrics[key].gauge > 0)
