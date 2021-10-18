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
    def test_session(self, mocked_upload_window, mocked_flush):
        now = int(time.time())
        session = graphsignal.session('d1')
        session.log_metadata(key='k1', value='v1')

        session.log_prediction(
            features={'A': 1, 'B': 2},
            prediction=False,
            actual_timestamp=now)

        session.log_prediction(
            features={'A': 10, 'B': 20},
            prediction=True,
            actual_timestamp=now)

        session.log_ground_truth(
            label='c1',
            prediction='c1',
            prediction_timestamp=now,
            segments=[
                's1',
                's2'])
        session.log_ground_truth(
            label='c1',
            prediction='c2',
            prediction_timestamp=now,
            segments=[
                's1',
                's3'])

        try:
            raise Exception('ex1')
        except Exception as ex:
            session.log_exception(
                message=ex, extra_info={
                    'k1': 'v1'}, exc_info=True, actual_timestamp=now)

        graphsignal.sessions.upload_all(force=True)

        mocked_upload_window.assert_called_once()

        uploaded_window = mocked_upload_window.call_args[0][0]
        self.assertEqual(uploaded_window.num_predictions, 2)
        self.assertEqual(uploaded_window.num_ground_truths, 2)
        self.assertEqual(uploaded_window.model.metadata['k1'], 'v1')
        self.assertEqual(
            uploaded_window.exceptions[0].message,
            'Exception: ex1')
        self.assertEqual(
            uploaded_window.exceptions[0].extra_info, {
                'k1': 'v1'})
        self.assertEqual(uploaded_window.num_exceptions, 1)
        self.assertTrue(uploaded_window.exceptions[0].stack_trace)
        self.assertEqual(len(uploaded_window.data_streams), 3)

    @patch.object(Uploader, 'flush')
    @patch.object(Uploader, 'upload_window')
    def test_prediction_not_uploaded(self, mocked_upload_window, mocked_flush):
        session = graphsignal.session(deployment_name='d1')

        session.log_prediction_batch(features=[[1, 2], [3, 4]])
        session.log_prediction_batch(features=[[1, 2], [3, 4]])
        session._current_window_updater._update_predictions()

        self.assertEqual(
            session._current_window_updater._window_proto.num_predictions, 4)
        mocked_upload_window.assert_not_called()

    @patch.object(Uploader, 'flush')
    @patch.object(Uploader, 'upload_window')
    def test_prediction_uploaded(
            self, mocked_upload_window, mocked_flush):
        session = graphsignal.session('d1')

        session.log_prediction_batch(
            features=[[1, 2], [3, 4]], actual_timestamp=int(time.time()) - 400)
        session.log_prediction_batch(
            features=[[1, 2], [3, 4]], actual_timestamp=int(time.time()))

        mocked_upload_window.assert_called_once()
