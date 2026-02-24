import unittest
import logging
import sys
import os
import time
from unittest.mock import patch, Mock
import pprint
import random

import graphsignal
from graphsignal.proto import signals_pb2

logger = logging.getLogger('graphsignal')

class LogStoreTest(unittest.TestCase):
    def setUp(self):
        graphsignal.configure(
            api_key='k1',
            debug_mode=True)
        graphsignal._ticker.auto_tick = False

    def tearDown(self):
        graphsignal.shutdown()

    @patch('time.time_ns', return_value=1)
    def test_update_and_export(self, mocked_time):
        store = graphsignal._ticker.log_store()
        store.clear()
        store.log_message(tags={'t1': '1'}, level='info', message='msg1', exception='exc1')
        store.log_message(tags={'t1': '1'}, level='info', message='msg2', exception='exc2')
        batches = store.export()

        # Entries with identical tags are grouped into a single LogBatch.
        self.assertEqual(len(batches), 1)

        first_batch = batches[0]
        tag_keys = {t.key: t.value for t in first_batch.tags}
        self.assertEqual(tag_keys.get('t1'), '1')

        # Both entries should be present in the batch.
        self.assertEqual(len(first_batch.log_entries), 2)
        first_entry = first_batch.log_entries[0]
        second_entry = first_batch.log_entries[1]

        self.assertEqual(first_entry.level, signals_pb2.LogEntry.LogLevel.INFO_LEVEL)
        self.assertEqual(first_entry.message, 'msg1')
        self.assertEqual(first_entry.exception, 'exc1')
        self.assertEqual(first_entry.log_ts, 1)

        self.assertEqual(second_entry.level, signals_pb2.LogEntry.LogLevel.INFO_LEVEL)
        self.assertEqual(second_entry.message, 'msg2')
        self.assertEqual(second_entry.exception, 'exc2')
        self.assertEqual(second_entry.log_ts, 1)

    def test_has_unexported(self):
        store = graphsignal._ticker.log_store()
        store.clear()
        self.assertFalse(store.has_unexported())
        store.log_message(tags={'t1': '1'}, level='info', message='msg1', exception='exc1')
        self.assertTrue(store.has_unexported())
