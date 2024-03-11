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

class LogStoreTest(unittest.TestCase):
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
        store = graphsignal._tracer.log_store()
        store.clear()
        store.log_message(scope='s1', name='m1', tags={'t1': '1'}, level='l1', message='msg1', exception='exc1')
        store.log_message(scope='s1', name='m1', tags={'t1': '1'}, level='l1', message='msg2', exception='exc2')
        protos = store.export()
        self.assertEqual(len(protos), 2)
        self.assertEqual(protos[0].scope, 's1')
        self.assertEqual(protos[0].name, 'm1')
        self.assertEqual(protos[0].tags[0].key, 't1')
        self.assertEqual(protos[0].tags[0].value, '1')
        self.assertEqual(protos[0].level, 'l1')
        self.assertEqual(protos[0].message, 'msg1')
        self.assertEqual(protos[0].exception, 'exc1')
        self.assertEqual(protos[0].create_ts, 1)

    def test_has_unexported(self):
        store = graphsignal._tracer.log_store()
        store.clear()
        self.assertFalse(store.has_unexported())
        store.log_message(scope='s1', name='m1', tags={'t1': '1'}, level='l1', message='msg1', exception='exc1')
        self.assertTrue(store.has_unexported())
