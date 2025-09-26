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

logger = logging.getLogger('graphsignal')

mem = []

def check_graphsignal():
    """Function to run in child process to check if graphsignal is configured"""
    try:
        import graphsignal
        has_tracer = hasattr(graphsignal, '_tracer') and graphsignal._tracer is not None
        has_correct_api_key = has_tracer and graphsignal._tracer.api_key == 'k1'
        return f"tracer_exists:{has_tracer},api_key_correct:{has_correct_api_key}"
    except Exception as e:
        return f"error:{str(e)}"

def worker_with_file(result_file):
    """Worker function that checks graphsignal and writes result to file"""
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
