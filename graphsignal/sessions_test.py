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
    @patch('platform.system', return_value='Linux')
    @patch.object(Uploader, 'flush')
    @patch.object(Uploader, 'upload_window')
    @patch('graphsignal.statistics.compute_metrics', return_value=[])
    @patch('graphsignal.system.vm_size', return_value=1)
    @patch('graphsignal.system.vm_rss', return_value=1)
    @patch('time.time', return_value=1)
    def test_prediction_batched(self, mocked_time, mocked_vm_rss, mocked_vm_size, mocked_compute_metrics,
                                mocked_upload_window, mocked_flush, mocked_system, mocked_version):
        session = graphsignal.session('m1', 'd1')
        session.set_attribute(name='a1', value='v1')
        session.log_metric(name='m1', value=1)
        session.log_event(
            description='e1', attributes={
                'a1': 'v1'}, is_error=True)
        session.log_prediction(input_data=[[1, 2], [3, 4]], output_data=[5, 6])
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
                         'name': 'user_defined_event',
                         'score': 1,
                         'timestamp': 1,
                         'type': 'error'}],
             'metrics': [{'dataset': 'user_defined',
                          'measurement': [1],
                          'name': 'm1',
                          'timestamp': 1,
                          'type': 'gauge',
                          'unit': ''},
                         {'dataset': 'system',
                          'measurement': [1],
                          'name': 'prediction_count',
                          'timestamp': 1,
                          'type': 'gauge',
                          'unit': ''},
                         {'dataset': 'system',
                          'measurement': [1],
                          'name': 'process_memory_usage',
                          'timestamp': 1,
                          'type': 'gauge',
                          'unit': 'KB'},
                         {'dataset': 'system',
                          'measurement': [1],
                          'name': 'process_virtual_memory',
                          'timestamp': 1,
                          'type': 'gauge',
                          'unit': 'KB'}],
             'model': {'attributes': [{'name': 'Python', 'value': '1.2.3'},
                                      {'name': 'OS', 'value': 'Linux'},
                                      {'name': 'a1', 'value': 'v1'}],
                       'name': 'm1',
                       'deployment': 'd1',
                       'timestamp': 1},
             'timestamp': 1}
        )

    @patch.object(Uploader, 'flush')
    @patch.object(Uploader, 'upload_window')
    def test_prediction_not_uploaded(self, mocked_upload_window, mocked_flush):
        session = graphsignal.session('m1')

        session.log_prediction(input_data=[[1, 2], [3, 4]])
        session.log_prediction(input_data=[[1, 2], [3, 4]])

        self.assertEqual(session._window_size, 4)
        mocked_upload_window.assert_not_called()

    @patch.object(Uploader, 'flush')
    @patch.object(Uploader, 'upload_window')
    def test_prediction_uploaded_max_window_duration(
            self, mocked_upload_window, mocked_flush):
        session = graphsignal.session('m1')

        session._window_start_time = session._window_start_time - \
            graphsignal.sessions.MAX_WINDOW_DURATION - 1
        session.log_prediction(input_data=[[1, 2], [3, 4]])

        mocked_upload_window.assert_called_once()

    @patch.object(Uploader, 'flush')
    @patch.object(Uploader, 'upload_window')
    def test_prediction_uploaded_min_window_size(
            self, mocked_upload_window, mocked_flush):
        session = graphsignal.session('m1')

        session._window_start_time = session._window_start_time - \
            graphsignal.sessions.MIN_WINDOW_DURATION - 1
        session.log_prediction(
            input_data=np.random.rand(
                graphsignal.sessions.MIN_WINDOW_SIZE, 2))

        mocked_upload_window.assert_called_once()

    @patch('platform.python_version', return_value='1.2.3')
    @patch('platform.system', return_value='Linux')
    @patch.object(Uploader, 'flush')
    @patch.object(Uploader, 'upload_window')
    @patch('graphsignal.system.vm_size', return_value=1)
    @patch('graphsignal.system.vm_rss', return_value=1)
    @patch('graphsignal.system.cpu_time', return_value=1)
    @patch('time.time', return_value=1)
    @patch('time.monotonic', return_value=1)
    def test_measure_latency_context(self, mocked_monotonic, mocked_time, mocked_cpu_time, mocked_vm_rss, mocked_vm_size,
                                     mocked_upload_window, mocked_flush, mocked_system, mocked_version):
        session = graphsignal.session('m1')

        try:
            with session.measure_latency():
                time.sleep(0.1)
                raise Exception('ex1')
        except BaseException:
            pass

        session._upload_window(force=True)

        window = mocked_upload_window.call_args[0][0]
        #pp = pprint.PrettyPrinter()
        # pp.pprint(window)

        self.assertEqual(
            window,
            {'metrics': [{'dataset': 'system',
                          'measurement': [0, 1],
                          'name': 'prediction_latency_p50',
                          'timestamp': 1,
                          'type': 'statistic',
                          'unit': 'ms'},
                         {'dataset': 'system',
                          'measurement': [0, 1],
                          'name': 'prediction_cpu_time_p50',
                          'timestamp': 1,
                          'type': 'statistic',
                          'unit': 'ms'},
                         {'dataset': 'system',
                          'measurement': [0],
                          'name': 'prediction_count',
                          'timestamp': 1,
                          'type': 'gauge',
                          'unit': ''},
                         {'dataset': 'system',
                          'measurement': [1],
                          'name': 'process_memory_usage',
                          'timestamp': 1,
                          'type': 'gauge',
                          'unit': 'KB'},
                         {'dataset': 'system',
                          'measurement': [1],
                          'name': 'process_virtual_memory',
                          'timestamp': 1,
                          'type': 'gauge',
                          'unit': 'KB'}],
             'model': {'attributes': [{'name': 'Python', 'value': '1.2.3'},
                                      {'name': 'OS', 'value': 'Linux'}],
                       'name': 'm1',
                       'timestamp': 1},
             'timestamp': 1}
        )

    @patch('platform.python_version', return_value='1.2.3')
    @patch('platform.system', return_value='Linux')
    @patch.object(Uploader, 'flush')
    @patch.object(Uploader, 'upload_window')
    @patch('graphsignal.system.vm_size', return_value=1)
    @patch('graphsignal.system.vm_rss', return_value=1)
    @patch('graphsignal.system.cpu_time', return_value=1)
    @patch('time.time', return_value=1)
    @patch('time.monotonic', return_value=1)
    def test_measure_latency_start_stop(
            self, mocked_monotonic, mocked_time, mocked_cpu_time, mocked_vm_rss, mocked_vm_size, mocked_upload_window, mocked_flush, mocked_system, mocked_version):
        session = graphsignal.session('m1')

        span = session.measure_latency()
        span.start()
        time.sleep(0.1)
        span.stop()

        session._upload_window(force=True)

        window = mocked_upload_window.call_args[0][0]
        #pp = pprint.PrettyPrinter()
        # pp.pprint(window)

        self.assertEqual(
            window,
            {'metrics': [{'dataset': 'system',
                          'measurement': [0, 1],
                          'name': 'prediction_latency_p50',
                          'timestamp': 1,
                          'type': 'statistic',
                          'unit': 'ms'},
                         {'dataset': 'system',
                          'measurement': [0, 1],
                          'name': 'prediction_cpu_time_p50',
                          'timestamp': 1,
                          'type': 'statistic',
                          'unit': 'ms'},
                         {'dataset': 'system',
                          'measurement': [0],
                          'name': 'prediction_count',
                          'timestamp': 1,
                          'type': 'gauge',
                          'unit': ''},
                         {'dataset': 'system',
                          'measurement': [1],
                          'name': 'process_memory_usage',
                          'timestamp': 1,
                          'type': 'gauge',
                          'unit': 'KB'},
                         {'dataset': 'system',
                          'measurement': [1],
                          'name': 'process_virtual_memory',
                          'timestamp': 1,
                          'type': 'gauge',
                          'unit': 'KB'}],
             'model': {'attributes': [{'name': 'Python', 'value': '1.2.3'},
                                      {'name': 'OS', 'value': 'Linux'}],
                       'name': 'm1',
                       'timestamp': 1},
             'timestamp': 1}
        )
