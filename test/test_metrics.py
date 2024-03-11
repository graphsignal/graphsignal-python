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
    def test_update_and_export(self, mocked_time):
        store = graphsignal._tracer.metric_store()
        store.set_gauge(scope='s1', name='m1', tags={'t1': '1'}, value=1, update_ts=10, unit='u1', is_time=True, is_size=True)
        store.set_gauge(scope='s1', name='m1', tags={'t1': '1'}, value=2, update_ts=10, unit='u1', is_time=True, is_size=True)
        protos = store.export()
        self.assertEqual(len(protos), 1)
        self.assertEqual(protos[0].scope, 's1')
        self.assertEqual(protos[0].name, 'm1')
        self.assertEqual(protos[0].tags[0].key, 't1')
        self.assertEqual(protos[0].tags[0].value, '1')
        self.assertEqual(protos[0].unit, 'u1')
        self.assertEqual(protos[0].is_time, True)
        self.assertEqual(protos[0].is_size, True)
        self.assertEqual(protos[0].gauge, 2)
        self.assertEqual(protos[0].update_ts, 10)

        store.clear()

        store.inc_counter(scope='s1', name='m1', tags={'t1': '1'}, value=1, update_ts=10, unit='u1')
        store.inc_counter(scope='s1', name='m1', tags={'t1': '1'}, value=2, update_ts=10, unit='u1')
        protos = store.export()
        self.assertEqual(len(protos), 1)
        self.assertEqual(protos[0].scope, 's1')
        self.assertEqual(protos[0].name, 'm1')
        self.assertEqual(protos[0].tags[0].key, 't1')
        self.assertEqual(protos[0].tags[0].value, '1')
        self.assertEqual(protos[0].unit, 'u1')
        self.assertEqual(protos[0].is_time, False)
        self.assertEqual(protos[0].is_size, False)
        self.assertEqual(protos[0].counter, 3)
        self.assertEqual(protos[0].update_ts, 10)

        store.clear()

        store.update_histogram(scope='s1', name='m1', tags={'t1': '1'}, value=1, update_ts=10, unit='u1', is_time=True, is_size=True)
        store.update_histogram(scope='s1', name='m1', tags={'t1': '1'}, value=2, update_ts=10, unit='u1', is_time=True, is_size=True)
        protos = store.export()
        self.assertEqual(len(protos), 1)
        self.assertEqual(protos[0].scope, 's1')
        self.assertEqual(protos[0].name, 'm1')
        self.assertEqual(protos[0].tags[0].key, 't1')
        self.assertEqual(protos[0].tags[0].value, '1')
        self.assertEqual(protos[0].unit, 'u1')
        self.assertEqual(protos[0].is_time, True)
        self.assertEqual(protos[0].is_size, True)
        self.assertEqual(protos[0].histogram.bins, [1, 2])
        self.assertEqual(protos[0].histogram.counts, [1, 1])
        self.assertEqual(protos[0].update_ts, 10)

    def test_has_unexported(self):
        store = graphsignal._tracer.metric_store()
        self.assertFalse(store.has_unexported())
        store.set_gauge(scope='s1', name='m1', tags={'t1': '1'}, value=1, update_ts=10, unit='u1', is_time=True, is_size=True)
        self.assertTrue(store.has_unexported())
