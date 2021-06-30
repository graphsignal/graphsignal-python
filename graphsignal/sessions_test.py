import unittest
import logging
import time
from unittest.mock import patch, Mock
import pprint
import numpy as np

import graphsignal
from graphsignal.uploader import Uploader
logger = logging.getLogger('graphsignal')


class SessionsTest(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None
        graphsignal.sessions.reset_all()
        graphsignal.configure(api_key='k1', debug_mode=True)

    def tearDown(self):
        graphsignal.sessions.reset_all()
        graphsignal.shutdown()

    @patch('platform.python_version', return_value='1.2.3')
    @patch.object(Uploader, 'flush')
    @patch.object(Uploader, 'upload_window')
    @patch('graphsignal.statistics.compute_metrics', return_value=[])
    @patch('time.time', return_value=1)
    def test_prediction_batched(self, mocked_time, mocked_compute_metrics,
                                mocked_upload_window, mocked_flush, mocked_version):
        session = graphsignal.session('d1')
        session.set_tag(name='t1', value='v1')
        session.log_prediction(input_data=[[1, 2], [3, 4]], output_data=[5, 6])
        session.log_event(
            description='e1', attributes={
                'a1': 'v1'}, is_error=True)

        session._upload_window(force=True)

        mocked_upload_window.assert_called_once()

        call_args = mocked_upload_window.call_args[0][0]
        call_args['metrics'] = sorted(
            call_args['metrics'], key=lambda metric: metric['name'])
        #pp = pprint.PrettyPrinter()
        # pp.pprint(call_args)

        self.assertEqual(
            call_args,
            {'events': [{'attributes': [{'name': 'a1', 'value': 'v1'}],
                        'description': 'e1',
                         'name': 'error',
                         'score': 1,
                         'timestamp': 1,
                         'type': 'error'}],
             'metrics': [{'aggregation': 'sum',
                         'dataset': 'model_statistics',
                          'measurement': [1],
                          'name': 'prediction_count',
                          'timestamp': 1,
                          'type': 'gauge',
                          'unit': ''}],
             'model': {'deployment': 'd1',
                       'tags': [{'name': 't1', 'value': 'v1'}],
                       'timestamp': 1},
             'timestamp': 1}
        )

    @patch.object(Uploader, 'flush')
    @patch.object(Uploader, 'upload_window')
    def test_prediction_not_uploaded(self, mocked_upload_window, mocked_flush):
        session = graphsignal.session(deployment_name='d1')

        session.log_prediction(input_data=[[1, 2], [3, 4]])
        session.log_prediction(input_data=[[1, 2], [3, 4]])

        self.assertEqual(session._window_size, 4)
        mocked_upload_window.assert_not_called()

    @patch.object(Uploader, 'flush')
    @patch.object(Uploader, 'upload_window')
    def test_prediction_uploaded_max_window_duration(
            self, mocked_upload_window, mocked_flush):
        session = graphsignal.session('d1')

        session._window_start_time = session._window_start_time - \
            graphsignal.sessions.MAX_WINDOW_DURATION - 1
        session.log_prediction(input_data=[[1, 2], [3, 4]])

        mocked_upload_window.assert_called_once()

    @patch.object(Uploader, 'flush')
    @patch.object(Uploader, 'upload_window')
    def test_prediction_uploaded_min_window_size(
            self, mocked_upload_window, mocked_flush):
        session = graphsignal.session('d1')

        session._window_start_time = session._window_start_time - \
            graphsignal.sessions.MIN_WINDOW_DURATION - 1
        session.log_prediction(
            input_data=np.random.rand(
                graphsignal.sessions.MIN_WINDOW_SIZE, 2))

        mocked_upload_window.assert_called_once()

    @patch.object(Uploader, 'flush')
    @patch.object(Uploader, 'upload_window')
    def test_session_with(self, mocked_upload_window, mocked_flush):
        with graphsignal.session(deployment_name='d1') as sess:
            self.assertEqual(sess._window_size, 0)

    @patch.object(Uploader, 'flush')
    @patch.object(Uploader, 'upload_window')
    def test_session_contextmanager(self, mocked_upload_window, mocked_flush):
        try:
            with graphsignal.session(deployment_name='d1') as sess:
                raise Exception('ex1')
        except BaseException:
            pass

        self.assertEqual(
            graphsignal.session(
                deployment_name='d1')._event_window[0].description,
            'Prediction exception')
