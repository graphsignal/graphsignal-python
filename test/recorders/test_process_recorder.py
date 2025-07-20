import unittest
import logging
import sys
from unittest.mock import patch, Mock
import socket
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
            debug_mode=True)
        graphsignal._tracer.auto_export = False

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
        metric_tags =  graphsignal._tracer.tags.copy()
        key = store.metric_key('process.cpu.usage', metric_tags)
        self.assertTrue(store._metrics[key].gauge > 0)
        key = store.metric_key('process.memory.usage', metric_tags)
        self.assertTrue(store._metrics[key].gauge > 0)
        key = store.metric_key('process.memory.virtual', metric_tags)
        self.assertTrue(store._metrics[key].gauge > 0)
        key = store.metric_key('host.memory.usage', metric_tags)
        self.assertTrue(store._metrics[key].gauge > 0)
