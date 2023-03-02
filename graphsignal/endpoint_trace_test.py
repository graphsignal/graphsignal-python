import unittest
import logging
import sys
import time
import numpy as np
from unittest.mock import patch, Mock

import graphsignal
from graphsignal.proto import signals_pb2
from graphsignal.endpoint_trace import EndpointTrace, DEFAULT_OPTIONS
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
            deployment='d1',
            tags={'k1': 'v1', 'k2': 'v2'},
            upload_on_shutdown=False,
            debug_mode=True)

    def tearDown(self):
        graphsignal.shutdown()

    @patch.object(ProcessRecorder, 'on_trace_stop')
    @patch.object(Uploader, 'upload_signal')
    @patch('time.time', return_value=1000)
    def test_start_stop(self, mocked_time, mocked_upload_signal, mocked_process_on_trace_stop):
        graphsignal.set_tag('k3', 'v3')
        graphsignal.log_param('p1', 'v1')

        for i in range(10):
            trace = EndpointTrace(
                endpoint='ep1',
                tags={'k3': 'v33', 'k4': 4.0})
            trace.set_tag('k5', 'v5')
            trace.set_param('p2', 'v2')
            trace.set_data('input', np.asarray([[1, 2],[3, 4]]), extra_counts=dict(c1=1, c2=2))
            time.sleep(0.01)
            trace.stop()

        signal = mocked_upload_signal.call_args[0][0]

        self.assertEqual(signal.deployment_name, 'd1')
        self.assertEqual(signal.endpoint_name, 'ep1')
        self.assertTrue(signal.worker_id != '')
        self.assertTrue(signal.start_us > 0)
        self.assertTrue(signal.end_us > 0)
        self.assertEqual(signal.signal_type, signals_pb2.SignalType.SAMPLE_SIGNAL)
        self.assertTrue(signal.process_usage.start_ms > 0)
        self.assertEqual(signal.trace_metrics.call_count.gauge, 10.0)
        self.assertEqual(len(signal.trace_metrics.latency_us.reservoir.values), 10)
        self.assertEqual(signal.data_metrics[0].data_name, 'input')
        self.assertEqual(signal.data_metrics[0].metric_name, 'element_count')
        self.assertEqual(signal.data_metrics[0].metric.gauge, 40.0)
        self.assertEqual(signal.data_metrics[-2].metric_name, 'c1')
        self.assertEqual(signal.data_metrics[-2].metric.gauge, 10.0)
        self.assertEqual(signal.data_metrics[-1].metric_name, 'c2')
        self.assertEqual(signal.data_metrics[-1].metric.gauge, 20.0)
        self.assertEqual(signal.tags[0].key, 'k1')
        self.assertEqual(signal.tags[0].value, 'v1')
        self.assertTrue(not signal.tags[0].is_trace_level)
        self.assertEqual(signal.tags[1].key, 'k2')
        self.assertEqual(signal.tags[1].value, 'v2')
        self.assertEqual(signal.tags[2].key, 'k3')
        self.assertEqual(signal.tags[2].value, 'v33')
        self.assertEqual(signal.tags[3].key, 'k4')
        self.assertEqual(signal.tags[3].value, '4.0')
        self.assertEqual(signal.tags[4].key, 'k5')
        self.assertEqual(signal.tags[4].value, 'v5')
        self.assertTrue(signal.tags[4].is_trace_level)
        self.assertEqual(signal.params[0].name, 'p1')
        self.assertEqual(signal.params[0].value, 'v1')
        self.assertEqual(signal.params[1].name, 'p2')
        self.assertEqual(signal.params[1].value, 'v2')
        self.assertTrue(signal.params[1].is_trace_level)
        self.assertEqual(signal.trace_sample.trace_idx, 10)
        self.assertTrue(signal.trace_sample.latency_us > 0)
        self.assertEqual(signal.data_profile[0].data_name, 'input')
        self.assertEqual(signal.data_profile[0].shape, [2, 2])
        self.assertEqual(signal.data_profile[0].counts[-2].name, 'c1')
        self.assertEqual(signal.data_profile[0].counts[-2].count, 1)
        self.assertEqual(signal.data_profile[0].counts[-1].name, 'c2')
        self.assertEqual(signal.data_profile[0].counts[-1].count, 2)

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
        self.assertEqual(signal.trace_metrics.exception_count.gauge, 2.0)
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
        self.assertEqual(signal.trace_metrics.exception_count.gauge, 1)
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
        self.assertEqual(signal.trace_metrics.exception_count.gauge, 1)
        self.assertEqual(signal.exceptions[0].exc_type, 'Exception')
        self.assertEqual(signal.exceptions[0].message, 'ex2')
        self.assertNotEqual(signal.exceptions[0].stack_trace, '')

    @patch.object(Uploader, 'upload_signal')
    def test_outlier(self, mocked_upload_signal):
        for _ in range(500):
            with EndpointTrace(endpoint='ep1'):
                time.sleep(0.00001)
        with EndpointTrace(endpoint='ep1'):
            time.sleep(0.01)

        has_outliers = False
        for call_args in mocked_upload_signal.call_args_list:
            signal = call_args[0][0]
            if signal.signal_type == signals_pb2.SignalType.LATENCY_OUTLIER_SIGNAL:
                has_outliers = True
                break
        self.assertTrue(has_outliers)

    @patch.object(Uploader, 'upload_signal')
    def test_set_data(self, mocked_upload_signal):
        with EndpointTrace(endpoint='ep1') as trace:
            trace.set_data('d1', {'c1': 100, 'c2': None}, check_missing_values=True)

        signal = mocked_upload_signal.call_args[0][0]

        self.assertEqual(signal.endpoint_name, 'ep1')
        self.assertTrue(signal.worker_id != '')
        self.assertTrue(signal.start_us > 0)
        self.assertTrue(signal.end_us > 0)
        self.assertEqual(signal.signal_type, signals_pb2.SignalType.MISSING_VALUES_SIGNAL)

    @patch.object(Uploader, 'upload_signal')
    def test_spans(self, mocked_upload_signal):
        trace = EndpointTrace(endpoint='ep1')
        trace2 = EndpointTrace(endpoint='ep2')
        trace3 = EndpointTrace(endpoint='ep3')
        trace3.stop()
        trace2.stop()
        trace4 = EndpointTrace(endpoint='ep4')
        trace4.stop()
        trace.stop()

        signal = mocked_upload_signal.call_args[0][0]

        self.assertEqual(signal.endpoint_name, 'ep1')

        self.assertEqual(signal.root_span.name, 'ep1')
        self.assertTrue(signal.root_span.start_ns > 0)
        self.assertTrue(signal.root_span.end_ns > 0)

        self.assertEqual(signal.root_span.spans[0].name, 'ep2')
        self.assertTrue(signal.root_span.spans[0].is_endpoint)
        self.assertTrue(signal.root_span.spans[0].start_ns > signal.root_span.start_ns)
        self.assertTrue(signal.root_span.spans[0].end_ns < signal.root_span.end_ns)

        self.assertEqual(signal.root_span.spans[0].spans[0].name, 'ep3')
        self.assertTrue(signal.root_span.spans[0].spans[0].is_endpoint)
        self.assertTrue(signal.root_span.spans[0].spans[0].start_ns > signal.root_span.spans[0].start_ns)
        self.assertTrue(signal.root_span.spans[0].spans[0].end_ns < signal.root_span.spans[0].end_ns)

        self.assertEqual(signal.root_span.spans[1].name, 'ep4')
        self.assertTrue(signal.root_span.spans[1].start_ns > signal.root_span.spans[0].end_ns)
        self.assertTrue(signal.root_span.spans[1].end_ns < signal.root_span.end_ns)
