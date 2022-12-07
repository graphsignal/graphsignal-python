import unittest
import logging
import sys
import time
import numpy as np
from unittest.mock import patch, Mock

import graphsignal
from graphsignal.proto import signals_pb2
from graphsignal.endpoint_trace import EndpointTrace
from graphsignal.recorders.process_recorder import ProcessRecorder
from graphsignal.uploader import Uploader
from graphsignal.data.missing_value_detector import MissingValueDetector

logger = logging.getLogger('graphsignal')


class EndpointTraceTest(unittest.TestCase):
    def setUp(self):
        if len(logger.handlers) == 0:
            logger.addHandler(logging.StreamHandler(sys.stdout))
        graphsignal.configure(
            api_key='k1',
            deployment='s1',
            debug_mode=True)

    def tearDown(self):
        graphsignal.shutdown()

    @patch.object(ProcessRecorder, 'on_trace_stop')
    @patch.object(Uploader, 'upload_signal')
    @patch('time.time', return_value=1000)
    def test_start_stop(self, mocked_time, mocked_upload_signal, mocked_process_on_trace_stop):
        for i in range(10):
            trace = EndpointTrace(
                endpoint='ep1',
                tags={'k1': 'v2', 'k3': 3.0})
            trace.set_tag('k4', 'v4')
            trace.set_param('p1', 'v1')
            trace.set_data('input', np.asarray([[1, 2],[3, 4]]))
            time.sleep(0.01)
            trace.stop()

        signal = mocked_upload_signal.call_args[0][0]

        self.assertEqual(signal.deployment_name, 's1')
        self.assertEqual(signal.endpoint_name, 'ep1')
        self.assertTrue(signal.worker_id != '')
        self.assertTrue(signal.start_us > 0)
        self.assertTrue(signal.end_us > 0)
        self.assertEqual(signal.signal_type, signals_pb2.SignalType.SAMPLE_SIGNAL)
        self.assertTrue(signal.process_usage.start_ms > 0)
        self.assertEqual(len(signal.trace_metrics.call_count.counter.buckets), 1)
        self.assertEqual(len(signal.trace_metrics.latency_us.reservoir.values), 9)
        self.assertEqual(signal.data_metrics[0].data_name, 'input')
        self.assertEqual(signal.data_metrics[0].metric_name, 'element_count')
        self.assertEqual(len(signal.data_metrics[0].metric.counter.buckets), 1)
        self.assertEqual(signal.tags[0].key, 'k1')
        self.assertEqual(signal.tags[0].value, 'v2')
        self.assertEqual(signal.tags[1].key, 'k3')
        self.assertEqual(signal.tags[1].value, '3.0')
        self.assertEqual(signal.tags[2].key, 'k4')
        self.assertEqual(signal.tags[2].value, 'v4')
        self.assertEqual(signal.tags[2].key, 'k4')
        self.assertEqual(signal.tags[2].value, 'v4')
        self.assertEqual(signal.trace_sample.trace_idx, 10)
        self.assertTrue(signal.trace_sample.latency_us > 0)
        self.assertEqual(signal.trace_sample.params[0].name, 'p1')
        self.assertEqual(signal.trace_sample.params[0].value, 'v1')
        self.assertEqual(signal.data_stats[0].data_name, 'input')
        self.assertEqual(signal.data_stats[0].shape, [2, 2])

    @patch.object(ProcessRecorder, 'on_trace_start')
    @patch.object(Uploader, 'upload_signal')
    def test_start_exception(self, mocked_upload_signal, mocked_process_on_trace_start):
        mocked_process_on_trace_start.side_effect = Exception('ex1')
        trace = EndpointTrace(endpoint='ep1')
        trace.stop()

        signal = mocked_upload_signal.call_args[0][0]

        self.assertEqual(signal.endpoint_name, 'ep1')
        self.assertTrue(signal.worker_id != '')
        self.assertTrue(signal.start_us > 0)
        self.assertTrue(signal.end_us > 0)
        self.assertEqual(signal.signal_type, signals_pb2.SignalType.SAMPLE_SIGNAL)
        self.assertEqual(signal.agent_errors[0].message, 'ex1')
        self.assertNotEqual(signal.agent_errors[0].stack_trace, '')

    @patch.object(ProcessRecorder, 'on_trace_stop')
    @patch.object(Uploader, 'upload_signal')
    def test_agent_exception(self, mocked_upload_signal, mocked_process_on_trace_stop):
        mocked_process_on_trace_stop.side_effect = Exception('ex1')
        trace = EndpointTrace(
            endpoint='ep1')
        trace.stop()

        signal = mocked_upload_signal.call_args[0][0]

        self.assertEqual(signal.endpoint_name, 'ep1')
        self.assertTrue(signal.worker_id != '')
        self.assertTrue(signal.start_us > 0)
        self.assertTrue(signal.end_us > 0)
        self.assertEqual(signal.agent_errors[0].message, 'ex1')
        self.assertNotEqual(signal.agent_errors[0].stack_trace, '')

    @patch.object(Uploader, 'upload_signal')
    def test_inference_exception(self, mocked_upload_signal):

        for _ in range(2):
            try:
                with EndpointTrace(endpoint='ep1'):
                    raise Exception('ex1')
            except Exception as ex:
                if str(ex) != 'ex1':
                    raise ex

        self.assertEqual(mocked_upload_signal.call_count, 2)
        signal = mocked_upload_signal.call_args[0][0]

        self.assertEqual(signal.endpoint_name, 'ep1')
        self.assertTrue(signal.worker_id != '')
        self.assertTrue(signal.signal_id != '')
        self.assertTrue(signal.start_us > 0)
        self.assertTrue(signal.end_us > 0)
        self.assertEqual(signal.signal_type, signals_pb2.SignalType.EXCEPTION_SIGNAL)
        self.assertEqual(len(signal.trace_metrics.exception_count.counter.buckets), 1)
        self.assertEqual(signal.exceptions[0].exc_type, 'Exception')
        self.assertEqual(signal.exceptions[0].message, 'ex1')
        self.assertNotEqual(signal.exceptions[0].stack_trace, '')

    @patch.object(Uploader, 'upload_signal')
    def test_set_exception(self, mocked_upload_signal):
        trace = EndpointTrace(endpoint='ep1')
        try:
            raise Exception('ex2')
        except Exception as ex:
            trace.set_exception(ex)
        trace.stop()

        signal = mocked_upload_signal.call_args[0][0]

        self.assertEqual(signal.endpoint_name, 'ep1')
        self.assertTrue(signal.worker_id != '')
        self.assertTrue(signal.start_us > 0)
        self.assertTrue(signal.end_us > 0)
        self.assertEqual(signal.signal_type, signals_pb2.SignalType.EXCEPTION_SIGNAL)
        self.assertEqual(len(signal.trace_metrics.exception_count.counter.buckets), 1)
        self.assertEqual(signal.exceptions[0].exc_type, 'Exception')
        self.assertEqual(signal.exceptions[0].message, 'ex2')
        self.assertNotEqual(signal.exceptions[0].stack_trace, '')

    @patch.object(Uploader, 'upload_signal')
    def test_set_exception_true(self, mocked_upload_signal):
        trace = EndpointTrace(endpoint='ep1')
        try:
            raise Exception('ex2')
        except Exception as ex:
            trace.set_exception(exc_info=True)
        trace.stop()

        signal = mocked_upload_signal.call_args[0][0]

        self.assertEqual(signal.endpoint_name, 'ep1')
        self.assertTrue(signal.worker_id != '')
        self.assertTrue(signal.start_us > 0)
        self.assertTrue(signal.end_us > 0)
        self.assertEqual(signal.signal_type, signals_pb2.SignalType.EXCEPTION_SIGNAL)
        self.assertEqual(len(signal.trace_metrics.exception_count.counter.buckets), 1)
        self.assertEqual(signal.exceptions[0].exc_type, 'Exception')
        self.assertEqual(signal.exceptions[0].message, 'ex2')
        self.assertNotEqual(signal.exceptions[0].stack_trace, '')

    @patch.object(Uploader, 'upload_signal')
    def test_set_data(self, mocked_upload_signal):
        with EndpointTrace(endpoint='ep1') as trace:
            trace.set_data('d1', {'c1': 100, 'c2': None})

        signal = mocked_upload_signal.call_args[0][0]

        self.assertEqual(signal.endpoint_name, 'ep1')
        self.assertTrue(signal.worker_id != '')
        self.assertTrue(signal.start_us > 0)
        self.assertTrue(signal.end_us > 0)
        self.assertEqual(signal.signal_type, signals_pb2.SignalType.MISSING_VALUES_SIGNAL)
