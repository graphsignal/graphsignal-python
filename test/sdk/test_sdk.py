import os
import unittest
import logging
from unittest.mock import patch

import graphsignal
import graphsignal.sdk
from graphsignal.sdk.signal_uploader import SignalUploader
from graphsignal.sdk.config_loader import ConfigLoader

logger = logging.getLogger('graphsignal')


class SdkTest(unittest.TestCase):
    def setUp(self):
        graphsignal.sdk.configure(
            api_key='k1',
            debug_mode=True)
        graphsignal.sdk.sdk()._auto_tick = False

    def tearDown(self):
        graphsignal.sdk.shutdown()

    def test_run_uid_tag_set(self):
        tags = graphsignal.sdk.sdk().tags()
        self.assertIn('run.uid', tags)
        self.assertEqual(len(tags['run.uid']), 12)

    def test_workload_id_tag_from_configure(self):
        graphsignal.sdk.shutdown()
        graphsignal.sdk.configure(
            api_key='k1', debug_mode=True, workload_id='abc123def456')
        self.assertEqual(
            graphsignal.sdk.sdk().tags()['workload.id'],
            'abc123def456')

    def test_workload_id_tag_from_env(self):
        graphsignal.sdk.shutdown()
        os.environ['GRAPHSIGNAL_WORKLOAD_ID'] = 'env-workload-1'
        try:
            graphsignal.sdk.configure(api_key='k1', debug_mode=True)
            self.assertEqual(
                graphsignal.sdk.sdk().tags()['workload.id'],
                'env-workload-1')
        finally:
            os.environ.pop('GRAPHSIGNAL_WORKLOAD_ID', None)

    def test_workload_id_param_not_overwritten_by_tags(self):
        graphsignal.sdk.shutdown()
        graphsignal.sdk.configure(
            api_key='k1',
            tags={'workload.id': 'from-tags'},
            workload_id='from-param',
            debug_mode=True)
        self.assertEqual(
            graphsignal.sdk.sdk().tags()['workload.id'], 'from-tags')

    def test_run_uid_tag_not_overwritten(self):
        graphsignal.sdk.shutdown()
        graphsignal.sdk.configure(
            api_key='k1',
            tags={'run.uid': 'my-run-1'},
            debug_mode=True)
        self.assertEqual(
            graphsignal.sdk.sdk().tags()['run.uid'], 'my-run-1')

    def test_run_uid_tag_from_env(self):
        graphsignal.sdk.shutdown()
        os.environ['GRAPHSIGNAL_RUN_UID'] = 'e2e-test-run'
        try:
            graphsignal.sdk.configure(api_key='k1', debug_mode=True)
            self.assertEqual(
                graphsignal.sdk.sdk().tags()['run.uid'], 'e2e-test-run')
        finally:
            os.environ.pop('GRAPHSIGNAL_RUN_UID', None)

    def test_run_uid_env_not_overwritten_by_tags(self):
        graphsignal.sdk.shutdown()
        os.environ['GRAPHSIGNAL_RUN_UID'] = 'from-env'
        try:
            graphsignal.sdk.configure(
                api_key='k1',
                tags={'run.uid': 'from-tags'},
                debug_mode=True)
            self.assertEqual(
                graphsignal.sdk.sdk().tags()['run.uid'], 'from-tags')
        finally:
            os.environ.pop('GRAPHSIGNAL_RUN_UID', None)

    @patch.object(SignalUploader, 'upload_metric')
    @patch.object(SignalUploader, 'flush')
    @patch.object(ConfigLoader, 'update_config')
    def test_shutdown_upload(self, mocked_update_config, mocked_flush, mocked_upload_metric):
        graphsignal.sdk.shutdown()
        graphsignal.sdk.configure(
            api_key='k1',
            debug_mode=True)
        graphsignal.sdk.sdk().set_gauge(name='n1', tags={}, value=1, measurement_ts=1)
        graphsignal.sdk.shutdown()

        self.assertTrue(mocked_upload_metric.call_count > 0)

