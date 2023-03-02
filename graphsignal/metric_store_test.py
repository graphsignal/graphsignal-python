import unittest
import logging
import sys
import os
import time
from unittest.mock import patch, Mock
from google.protobuf.json_format import MessageToJson
import pprint
import random

import graphsignal
from graphsignal.proto import signals_pb2

logger = logging.getLogger('graphsignal')

class MetricStoreTest(unittest.TestCase):
    def setUp(self):
        graphsignal.configure(
            api_key='k1',
            deployment='d1',
            upload_on_shutdown=False,
            debug_mode=True)

    def tearDown(self):
        graphsignal.shutdown()

    @patch('time.time', return_value=1)
    def test_update(self, mocked_time):
        store = graphsignal._agent.metric_store('ep1')

        store.add_latency(20, 1000 * 1e6)
        store.add_latency(30, 1000 * 1e6)

        store.inc_call_count(1, 1000 * 1e6)
        store.inc_call_count(1, 1000 * 1e6)
        store.inc_call_count(1, 1001 * 1e6)
    
        store.inc_exception_count(1, 100 * 1e6)
        store.inc_exception_count(300, 1001 * 1e6)

        store.inc_data_counter('d1', 'c1', 1, 1000 * 1e6)
        store.inc_data_counter('d1', 'c1', 1, 1000 * 1e6)
        store.inc_data_counter('d1', 'c1', 1, 1001 * 1e6)
        store.inc_data_counter('d1', 'c2', 1, 1001 * 1e6)

        signal = signals_pb2.WorkerSignal()
        store.export(signal, 1001 * 1e6)

        #pp = pprint.PrettyPrinter()
        #pp.pprint(MessageToJson(signal))

        self.assertEqual(signal.trace_metrics.latency_us.reservoir.values, [20, 30])
        self.assertEqual(signal.trace_metrics.call_count.gauge, 1.5)
        self.assertEqual(signal.trace_metrics.exception_count.gauge, 0.5)
        self.assertEqual(signal.data_metrics[0].data_name, 'd1')
        self.assertEqual(signal.data_metrics[0].metric_name, 'c1')
        self.assertEqual(signal.data_metrics[0].metric.gauge, 1.5)
        self.assertEqual(signal.data_metrics[1].data_name, 'd1')
        self.assertEqual(signal.data_metrics[1].metric_name, 'c2')
        self.assertEqual(signal.data_metrics[1].metric.gauge, 1.0)

    def test_is_latency_outlier(self):
        store = graphsignal._agent.metric_store('ep1')

        for i in range(1000):
            val = random.randint(100, 200)
            now = int(time.time() / 1e6) + i
            store.add_latency(val, now)
            self.assertFalse(store.is_latency_outlier(val, now))
        self.assertTrue(store.is_latency_outlier(1000, now + 1))