import unittest
import logging
import sys
from unittest.mock import patch, Mock, MagicMock
import socket
import pprint
import time
import random

import graphsignal
from graphsignal.recorders.process_recorder import ProcessRecorder
from test.test_utils import find_last_datapoint

logger = logging.getLogger('graphsignal')

mem = []

def check_graphsignal():
    try:
        import graphsignal
        has_ticker = hasattr(graphsignal, '_ticker') and graphsignal._ticker is not None
        has_correct_api_key = has_ticker and graphsignal._ticker.api_key == 'k1'
        return f"ticker_exists:{has_ticker},api_key_correct:{has_correct_api_key}"
    except Exception as e:
        return f"error:{str(e)}"

def worker_with_file(result_file):
    result = check_graphsignal()
    with open(result_file, 'w') as f:
        f.write(result)

class ProcessRecorderTest(unittest.TestCase):
    def setUp(self):
        if len(logger.handlers) == 0:
            logger.addHandler(logging.StreamHandler(sys.stdout))
        graphsignal.configure(
            api_key='k1',
            debug_mode=True)
        graphsignal._ticker.auto_tick = False

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

        recorder.on_tick()

        store = graphsignal._ticker.metric_store()
        metric_tags =  graphsignal._ticker.tags.copy()
        key = store.metric_key('process.cpu.usage', metric_tags)
        self.assertTrue(find_last_datapoint(store, key).gauge > 0)
        key = store.metric_key('process.memory.usage', metric_tags)
        self.assertTrue(find_last_datapoint(store, key).gauge > 0)
        key = store.metric_key('process.memory.virtual', metric_tags)
        self.assertTrue(find_last_datapoint(store, key).gauge > 0)
        key = store.metric_key('host.memory.usage', metric_tags)
        self.assertTrue(find_last_datapoint(store, key).gauge > 0)
