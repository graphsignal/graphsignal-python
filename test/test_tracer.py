import unittest
import logging
import sys
import os
import time
from unittest.mock import patch, Mock
import pprint

import graphsignal
from graphsignal.uploader import Uploader

logger = logging.getLogger('graphsignal')

class TracerTest(unittest.TestCase):
    def setUp(self):
        graphsignal.configure(
            api_key='k1',
            deployment='d1',
            upload_on_shutdown=False,
            debug_mode=True)

    def tearDown(self):
        graphsignal.shutdown()

    @patch.object(Uploader, 'upload_metric')
    @patch.object(Uploader, 'upload_log_entry')
    def test_shutdown_upload(self, mocked_upload_log_entry, mocked_upload_metric):
        graphsignal.shutdown()
        graphsignal.configure(
            api_key='k1',
            deployment='d1',
            debug_mode=True)
        graphsignal._tracer.metric_store().set_gauge(scope='s1', name='n1', tags={}, value=1, update_ts=1)
        graphsignal.shutdown()

        model = mocked_upload_metric.call_args[0][0]

        self.assertEqual(model.scope, 's1')
        self.assertEqual(model.name, 'n1')

    @patch('graphsignal.tracer.uuid_sha1', return_value='123')
    def test_context_tag(self, mocked_uuid_sha1):
        tracer = graphsignal._tracer
        
        tracer.set_context_tag('k1', 'v1')
        self.assertEqual(tracer.get_context_tag('k1'), 'v1')

        tracer.set_context_tag('k2', 'v2', append_uuid=True)
        self.assertEqual(tracer.get_context_tag('k2'), 'v2-123')

        tracer.remove_context_tag('k1')
        self.assertEqual(tracer.get_context_tag('k1'), None)

        tracer.set_context_tag('k2', None)
        self.assertEqual(tracer.get_context_tag('k2'), None)
