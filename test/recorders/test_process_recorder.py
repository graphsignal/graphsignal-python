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
        has_correct_api_key = has_ticker and graphsignal._ticker.api_key() == 'k1'
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
        graphsignal._ticker._auto_tick = False

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
        metric_tags = graphsignal._ticker.process_tags()
        key = store.metric_key('process.cpu.usage', metric_tags)
        self.assertTrue(find_last_datapoint(store, key).gauge > 0)
        key = store.metric_key('process.memory.usage', metric_tags)
        self.assertTrue(find_last_datapoint(store, key).gauge > 0)
        key = store.metric_key('process.memory.virtual', metric_tags)
        self.assertTrue(find_last_datapoint(store, key).gauge > 0)
        key = store.metric_key('host.memory.usage', metric_tags)
        self.assertTrue(find_last_datapoint(store, key).gauge > 0)

        resource_store = graphsignal._ticker.resource_store()
        self.assertTrue(resource_store.has_unexported())
        resources = resource_store.export()

        process_resources = [r for r in resources if r.kind == 'process']
        self.assertEqual(len(process_resources), 1)
        process_attr_names = [a.name for a in process_resources[0].attributes]
        self.assertIn('process.command_line', process_attr_names)

        node_resources = [r for r in resources if r.kind == 'node']
        self.assertEqual(len(node_resources), 1)
        node_tag_dict = {t.key: t.value for t in node_resources[0].tags}
        self.assertIn('host.name', node_tag_dict)
        node_attr_names = [a.name for a in node_resources[0].attributes]
        self.assertIn('platform', node_attr_names)
        self.assertIn('machine', node_attr_names)
