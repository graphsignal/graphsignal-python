import unittest
import logging
import time
from unittest.mock import patch, Mock
import sys
import pprint
import numpy as np

import graphsignal
from graphsignal.uploader import Uploader
from graphsignal import metrics_pb2

logger = logging.getLogger('graphsignal')


class SessionsTest(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None
        if len(logger.handlers) == 0:
            logger.addHandler(logging.StreamHandler(sys.stdout))
        graphsignal.sessions.reset_all()
        graphsignal.configure(api_key='k1', debug_mode=True)

    def tearDown(self):
        graphsignal.sessions.reset_all()
        graphsignal.shutdown()

    @patch.object(Uploader, 'flush')
    @patch.object(Uploader, 'upload_window')
    @patch('graphsignal.statistics.update_metrics')
    def test_prediction_batched(self, mocked_update_metrics,
                                mocked_upload_window, mocked_flush):
        session = graphsignal.session('d1')
        session.set_metadata(key='k1', value='v1')
        session.log_prediction(
            input_data=[[1, 2], [3, 4]],
            input_columns=['A', 'B'],
            output_data=[5, 6])

        try:
            raise Exception('ex1')
        except Exception as ex:
            session.log_exception(message=ex, extra_info={'k1': 'v1'}, exc_info=True)

        session._upload_window(force=True)

        mocked_upload_window.assert_called_once()

        uploaded_window = mocked_upload_window.call_args[0][0]
        self.assertEqual(uploaded_window.num_predictions, 2)
        self.assertEqual(uploaded_window.model.metadata['k1'], 'v1')
        self.assertEqual(
            uploaded_window.exceptions[0].message,
            'Exception: ex1')
        self.assertEqual(
            uploaded_window.exceptions[0].extra_info, {
                'k1': 'v1'})
        self.assertTrue(uploaded_window.exceptions[0].stack_trace)

    @patch.object(Uploader, 'flush')
    @patch.object(Uploader, 'upload_window')
    def test_prediction_not_uploaded(self, mocked_upload_window, mocked_flush):
        session = graphsignal.session(deployment_name='d1')

        session.log_prediction(input_data=[[1, 2], [3, 4]])
        session.log_prediction(input_data=[[1, 2], [3, 4]])

        self.assertEqual(session._current_window.num_predictions, 4)
        mocked_upload_window.assert_not_called()

    @patch.object(Uploader, 'flush')
    @patch.object(Uploader, 'upload_window')
    def test_prediction_uploaded(
            self, mocked_upload_window, mocked_flush):
        session = graphsignal.session('d1')

        session.log_prediction(input_data=[[1, 2], [3, 4]])
        session._current_window.start_ts = session._current_window.start_ts - \
            graphsignal._get_config().window_seconds - 1
        session.log_prediction(input_data=[[1, 2], [3, 4]])

        mocked_upload_window.assert_called_once()
