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
            debug_mode=True)
        graphsignal._tracer.export_on_shutdown = False

    def tearDown(self):
        graphsignal.shutdown()

    @patch.object(Uploader, 'upload_metric')
    @patch.object(Uploader, 'upload_log_entry')
    def test_shutdown_upload(self, mocked_upload_log_entry, mocked_upload_metric):
        graphsignal.shutdown()
        graphsignal.configure(
            api_key='k1',
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

    def test_set_profiling_mode_success(self):
        tracer = graphsignal._tracer
        tracer.profiling_rate = 1
        result = tracer.set_profiling_mode()
        self.assertTrue(result)
        result = tracer.set_profiling_mode()
        self.assertFalse(result)

    def test_set_profiling_mode_fail(self):
        tracer = graphsignal._tracer
        tracer.profiling_rate = 0.0
        result = tracer.set_profiling_mode()
        self.assertFalse(result)

    def test_set_profiling_mode_already_set_not_expired(self):
        tracer = graphsignal._tracer
        tracer.profiling_rate = 1.0
        tracer._profiling_mode = time.time()
        result = tracer.set_profiling_mode()
        self.assertFalse(result)

    def test_set_profiling_mode_expired(self):
        tracer = graphsignal._tracer
        tracer.profiling_rate = 1.0
        tracer._profiling_mode = time.time() - (tracer.PROFILING_MODE_TIMEOUT_SEC + 1)
        result = tracer.set_profiling_mode()
        self.assertTrue(result)

    def test_unset_profiling_mode(self):
        tracer = graphsignal._tracer
        tracer.profiling_rate = 1.0
        tracer.set_profiling_mode()
        self.assertTrue(tracer.is_profiling_mode())
        tracer.unset_profiling_mode()
        self.assertFalse(tracer.is_profiling_mode())