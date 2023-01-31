import unittest
import logging
import sys
import os
import time
from unittest.mock import patch, Mock
from google.protobuf.json_format import MessageToJson
import pprint

import graphsignal
from graphsignal.proto import signals_pb2

logger = logging.getLogger('graphsignal')

class MetricStoreTest(unittest.TestCase):
    def setUp(self):
        graphsignal.configure(
            api_key='k1',
            deployment='d1',
            debug_mode=True)

    def tearDown(self):
        graphsignal.shutdown()

    @patch('time.time', return_value=1)
    def test_update_metric_store(self, mocked_time):
        store = graphsignal._agent.metric_store('ep1')

        store.add_time(20)
        store.add_time(30)

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
        store.convert_to_proto(signal, 1001 * 1e6)

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
