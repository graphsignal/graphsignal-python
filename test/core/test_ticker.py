import unittest
import logging
import time
from unittest.mock import patch, Mock
import pprint

import graphsignal
from graphsignal.core.signal_uploader import SignalUploader
from graphsignal.core.config_loader import ConfigLoader
from graphsignal.signals.spans import Span
from test.test_utils import find_tag

logger = logging.getLogger('graphsignal')

class TickerTest(unittest.TestCase):
    def setUp(self):
        graphsignal.configure(
            api_key='k1',
            debug_mode=True)
        graphsignal._ticker.auto_tick = False

    def tearDown(self):
        graphsignal.shutdown()

    @patch.object(SignalUploader, 'upload_metric')
    @patch.object(SignalUploader, 'flush')
    @patch.object(ConfigLoader, 'update_config')
    def test_shutdown_upload(self, mocked_update_config, mocked_flush, mocked_upload_metric):
        graphsignal.shutdown()
        graphsignal.configure(
            api_key='k1',
            debug_mode=True)
        graphsignal._ticker.set_gauge(name='n1', tags={}, value=1, measurement_ts=1)
        graphsignal.shutdown()

        self.assertTrue(mocked_upload_metric.call_count > 0)

    @patch('graphsignal.core.ticker.uuid_sha1', return_value='123')
    def test_context_tag(self, mocked_uuid_sha1):
        ticker = graphsignal._ticker
        
        ticker.set_context_tag('k1', 'v1')
        self.assertEqual(ticker.get_context_tag('k1'), 'v1')

        ticker.set_context_tag('k2', 'v2', append_uuid=True)
        self.assertEqual(ticker.get_context_tag('k2'), 'v2-123')

        ticker.remove_context_tag('k1')
        self.assertEqual(ticker.get_context_tag('k1'), None)

        ticker.set_context_tag('k2', None)
        self.assertEqual(ticker.get_context_tag('k2'), None)

    def test_sampler_initialization(self):
        ticker = graphsignal._ticker
        self.assertIsNotNone(ticker._samplers)
        self.assertEqual(len(ticker._samplers), 0)  # Initially empty

    def test_debug_mode_update_from_config_loader(self):
        # Start with debug_mode disabled at configure-time.
        graphsignal.shutdown()
        graphsignal.configure(
            api_key='k1',
            debug_mode=False)
        graphsignal._ticker.auto_tick = False

        ticker = graphsignal._ticker
        self.assertIs(ticker.debug_mode, False)

        # Ensure we propagate to profilers too.
        ticker._cupti_profiler.set_debug_mode = Mock()

        # Enable via config_loader option and emit update.
        ticker.config_loader()._options['debug_mode'] = '1'
        ticker.config_loader().emit_update(['debug_mode'])

        self.assertIs(ticker.debug_mode, True)
        self.assertEqual(logger.level, logging.DEBUG)
        ticker._cupti_profiler.set_debug_mode.assert_called_with(True)

        # Disable via config_loader option and emit update.
        ticker._cupti_profiler.set_debug_mode.reset_mock()
        ticker.config_loader()._options['debug_mode'] = '0'
        ticker.config_loader().emit_update(['debug_mode'])

        self.assertIs(ticker.debug_mode, False)
        self.assertEqual(logger.level, logging.WARNING)
        ticker._cupti_profiler.set_debug_mode.assert_called_with(False)

