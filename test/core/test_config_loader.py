import unittest
import sys
import logging
import json

from unittest.mock import patch, Mock

import graphsignal
from graphsignal.core.config_loader import ConfigLoader
from test.http_server import HttpTestServer

logger = logging.getLogger('graphsignal')


class ConfigLoaderTest(unittest.TestCase):
    def setUp(self):
        if len(logger.handlers) == 0:
            logger.addHandler(logging.StreamHandler(sys.stdout))
        graphsignal.configure(
            api_key='k1',
            debug_mode=True)
        graphsignal._ticker.auto_tick = False
        graphsignal._ticker.config_loader().clear()

    def tearDown(self):
        graphsignal._ticker.config_loader().clear()
        graphsignal.shutdown()

    @patch('requests.get')
    def test_update_config(self, mocked_get):
        mock_response = Mock()
        mock_response.json.return_value = {
            'options': [
                {'name': 'traces_per_sec', 'value': '2.5'},
                {'name': 'some_opt', 'value': '10000000'}
            ]
        }
        mock_response.raise_for_status = Mock()
        mocked_get.return_value = mock_response

        graphsignal._ticker.config_loader().update_config()

        mocked_get.assert_called_once()
        self.assertEqual(graphsignal._ticker.config_loader().get_float_option('traces_per_sec'), 2.5)
        self.assertEqual(graphsignal._ticker.config_loader().get_int_option('some_opt'), 10_000_000)

    @patch('requests.get')
    def test_update_config_defaults(self, mocked_get):
        mock_response = Mock()
        mock_response.json.return_value = {
            'options': [
                {'name': 'traces_per_sec', 'value': '0'},
                {'name': 'some_opt', 'value': '0'}
            ]
        }
        mock_response.raise_for_status = Mock()
        mocked_get.return_value = mock_response

        graphsignal._ticker.config_loader().update_config()

        self.assertEqual(graphsignal._ticker.config_loader().get_float_option('traces_per_sec'), 0.0)
        self.assertEqual(graphsignal._ticker.config_loader().get_int_option('some_opt'), 0)

    @patch('requests.get')
    def test_update_config_fail(self, mocked_get):
        def side_effect(*args, **kwargs):
            raise Exception("Ex1")
        mocked_get.side_effect = side_effect

        graphsignal._ticker.config_loader().update_config()

        mocked_get.assert_called_once()
        self.assertIsNone(graphsignal._ticker.config_loader().get_float_option('traces_per_sec'))
        self.assertIsNone(graphsignal._ticker.config_loader().get_int_option('some_opt'))

    def test_get_config(self):
        graphsignal._ticker.api_url = 'http://localhost:5006'

        server = HttpTestServer(5006)
        response_data = {
            'options': [
                {'name': 'traces_per_sec', 'value': '1.5'},
                {'name': 'some_opt', 'value': '10000000'}
            ]
        }
        server.set_response_data(json.dumps(response_data).encode('utf-8'))
        server.start()
        server.wait_ready()

        graphsignal._ticker.config_loader().update_config()

        server.join(timeout=2.0)

        self.assertEqual(graphsignal._ticker.config_loader().get_float_option('traces_per_sec'), 1.5)
        self.assertEqual(graphsignal._ticker.config_loader().get_int_option('some_opt'), 10_000_000)

    @patch('requests.get')
    def test_update_callback_changed_options(self, mocked_get):
        callback_calls = []

        def update_callback(changed_options):
            callback_calls.append(changed_options.copy())

        graphsignal._ticker.config_loader().on_update(update_callback)

        # First update - all options are new
        mock_response1 = Mock()
        mock_response1.json.return_value = {
            'options': [
                {'name': 'traces_per_sec', 'value': '2.5'},
                {'name': 'some_opt', 'value': '10000000'},
                {'name': 'some_other_opt', 'value': 'value1'}
            ]
        }
        mock_response1.raise_for_status = Mock()
        mocked_get.return_value = mock_response1

        graphsignal._ticker.config_loader().update_config()

        # Should have been called with all options (they're all new)
        self.assertEqual(len(callback_calls), 1)
        self.assertEqual(set(callback_calls[0]), {'traces_per_sec', 'some_opt', 'some_other_opt'})

        # Second update - only one option changes
        callback_calls.clear()
        mock_response2 = Mock()
        mock_response2.json.return_value = {
            'options': [
                {'name': 'traces_per_sec', 'value': '3.0'},  # Changed
                {'name': 'some_opt', 'value': '10000000'},  # Unchanged
                {'name': 'some_other_opt', 'value': 'value1'}  # Unchanged
            ]
        }
        mock_response2.raise_for_status = Mock()
        mocked_get.return_value = mock_response2

        graphsignal._ticker.config_loader().update_config()

        # Should have been called with only the changed option
        self.assertEqual(len(callback_calls), 1)
        self.assertEqual(callback_calls[0], ['traces_per_sec'])

        # Third update - multiple options change
        callback_calls.clear()
        mock_response3 = Mock()
        mock_response3.json.return_value = {
            'options': [
                {'name': 'traces_per_sec', 'value': '4.0'},  # Changed
                {'name': 'some_opt', 'value': '20000000'},  # Changed
                {'name': 'some_other_opt', 'value': 'value1'}  # Unchanged
            ]
        }
        mock_response3.raise_for_status = Mock()
        mocked_get.return_value = mock_response3

        graphsignal._ticker.config_loader().update_config()

        # Should have been called with both changed options
        self.assertEqual(len(callback_calls), 1)
        self.assertEqual(set(callback_calls[0]), {'traces_per_sec', 'some_opt'})

        # Fourth update - no changes
        callback_calls.clear()
        mock_response4 = Mock()
        mock_response4.json.return_value = {
            'options': [
                {'name': 'traces_per_sec', 'value': '4.0'},  # Unchanged
                {'name': 'some_opt', 'value': '20000000'},  # Unchanged
                {'name': 'some_other_opt', 'value': 'value1'}  # Unchanged
            ]
        }
        mock_response4.raise_for_status = Mock()
        mocked_get.return_value = mock_response4

        graphsignal._ticker.config_loader().update_config()

        # Should not have been called (no changes)
        self.assertEqual(len(callback_calls), 0)

        # Fifth update - option removed
        callback_calls.clear()
        mock_response5 = Mock()
        mock_response5.json.return_value = {
            'options': [
                {'name': 'traces_per_sec', 'value': '4.0'},  # Unchanged
                {'name': 'some_opt', 'value': '20000000'}  # Unchanged
                # some_other_opt removed
            ]
        }
        mock_response5.raise_for_status = Mock()
        mocked_get.return_value = mock_response5

        graphsignal._ticker.config_loader().update_config()

        # Should have been called with the removed option
        self.assertEqual(len(callback_calls), 1)
        self.assertEqual(callback_calls[0], ['some_other_opt'])


