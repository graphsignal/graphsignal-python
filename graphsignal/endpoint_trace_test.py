import unittest
import logging
import sys
import time
import numpy as np
from unittest.mock import patch, Mock

import graphsignal
from graphsignal.proto import signals_pb2
from graphsignal.endpoint_trace import EndpointTrace
from graphsignal.profilers.tensorflow import TensorFlowProfiler
from graphsignal.usage.process_reader import ProcessReader
from graphsignal.usage.nvml_reader import NvmlReader
from graphsignal.uploader import Uploader
from graphsignal.data.missing_value_detector import MissingValueDetector

logger = logging.getLogger('graphsignal')


class TracerTest(unittest.TestCase):
    def setUp(self):
        if len(logger.handlers) == 0:
            logger.addHandler(logging.StreamHandler(sys.stdout))
        graphsignal.configure(
            api_key='k1',
            deployment='s1',
            debug_mode=True)

    def tearDown(self):
        graphsignal.shutdown()

    @patch.object(TensorFlowProfiler, 'start', return_value=True)
    @patch.object(TensorFlowProfiler, 'stop', return_value=True)
    @patch.object(ProcessReader, 'read')
    @patch.object(NvmlReader, 'read')
    @patch.object(Uploader, 'upload_signal')
    @patch('time.time', return_value=1000)
    def test_start_stop(self, mocked_time, mocked_upload_signal, mocked_nvml_read, mocked_host_read,
                        mocked_stop, mocked_start):
        for i in range(10):
            trace = EndpointTrace(
                endpoint='ep1',
                tags={'k1': 'v2', 'k3': 3.0},
                profiler=TensorFlowProfiler())
            trace.set_tag('k4', 'v4')
            trace.set_data('input', np.asarray([[1, 2],[3, 4]]))
            time.sleep(0.01)
            trace.stop()

        self.assertEqual(mocked_start.call_count, 2)
        self.assertEqual(mocked_stop.call_count, 2)
        signal = mocked_upload_signal.call_args[0][0]

        self.assertEqual(signal.deployment_name, 's1')
        self.assertEqual(signal.endpoint_name, 'ep1')
        self.assertTrue(signal.worker_id != '')
        self.assertTrue(signal.start_us > 0)
        self.assertTrue(signal.end_us > 0)
        self.assertEqual(signal.signal_type, signals_pb2.SignalType.PROFILE_SIGNAL)
        self.assertEqual(len(signal.trace_metrics.call_count.counter.buckets), 1)
        self.assertEqual(len(signal.trace_metrics.latency_us.reservoir.values), 8)
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
        self.assertEqual(signal.data_stats[0].data_name, 'input')
        self.assertEqual(signal.data_stats[0].shape, [2, 2])

    @patch.object(TensorFlowProfiler, 'start', return_value=True)
    @patch.object(TensorFlowProfiler, 'stop', return_value=True)
    @patch.object(ProcessReader, 'read')
    @patch.object(NvmlReader, 'read')
    @patch.object(Uploader, 'upload_signal')
    def test_start_exception(self, mocked_upload_signal, mocked_nvml_read, mocked_host_read,
                             mocked_stop, mocked_start):
        mocked_start.side_effect = Exception('ex1')
        trace = EndpointTrace(
            endpoint='ep1',
            profiler=TensorFlowProfiler())
        trace.stop()

        mocked_start.assert_called_once()
        mocked_stop.assert_not_called()
        signal = mocked_upload_signal.call_args[0][0]

        self.assertEqual(signal.endpoint_name, 'ep1')
        self.assertTrue(signal.worker_id != '')
        self.assertTrue(signal.start_us > 0)
        self.assertTrue(signal.end_us > 0)
        self.assertEqual(signal.signal_type, signals_pb2.SignalType.SAMPLE_SIGNAL)
        self.assertEqual(signal.profiler_errors[0].message, 'ex1')
        self.assertNotEqual(signal.profiler_errors[0].stack_trace, '')

    @patch.object(TensorFlowProfiler, 'start', return_value=True)
    @patch.object(TensorFlowProfiler, 'stop', return_value=True)
    @patch.object(ProcessReader, 'read')
    @patch.object(NvmlReader, 'read')
    @patch.object(Uploader, 'upload_signal')
    def test_profiler_exception(self, mocked_upload_signal, mocked_nvml_read, mocked_host_read,
                            mocked_stop, mocked_start):
        mocked_stop.side_effect = Exception('ex1')
        trace = EndpointTrace(
            endpoint='ep1',
            profiler=TensorFlowProfiler())
        trace.stop()

        mocked_start.assert_called_once()
        mocked_stop.assert_called_once()
        signal = mocked_upload_signal.call_args[0][0]

        self.assertEqual(signal.endpoint_name, 'ep1')
        self.assertTrue(signal.worker_id != '')
        self.assertTrue(signal.start_us > 0)
        self.assertTrue(signal.end_us > 0)
        self.assertEqual(signal.profiler_errors[0].message, 'ex1')
        self.assertNotEqual(signal.profiler_errors[0].stack_trace, '')

    @patch.object(TensorFlowProfiler, 'start', return_value=True)
    @patch.object(TensorFlowProfiler, 'stop', return_value=True)
    @patch.object(ProcessReader, 'read')
    @patch.object(NvmlReader, 'read')
    @patch.object(Uploader, 'upload_signal')
    def test_inference_exception(self, mocked_upload_signal, mocked_nvml_read, mocked_host_read,
                            mocked_stop, mocked_start):

        for _ in range(2):
            try:
                with EndpointTrace(endpoint='ep1', profiler=TensorFlowProfiler()):
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

    @patch.object(TensorFlowProfiler, 'start', return_value=True)
    @patch.object(TensorFlowProfiler, 'stop', return_value=True)
    @patch.object(ProcessReader, 'read')
    @patch.object(NvmlReader, 'read')
    @patch.object(Uploader, 'upload_signal')
    def test_set_exception(self, mocked_upload_signal, mocked_nvml_read, mocked_host_read,
                            mocked_stop, mocked_start):
        trace = EndpointTrace(endpoint='ep1', profiler=TensorFlowProfiler())
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

    @patch.object(TensorFlowProfiler, 'start', return_value=True)
    @patch.object(TensorFlowProfiler, 'stop', return_value=True)
    @patch.object(ProcessReader, 'read')
    @patch.object(NvmlReader, 'read')
    @patch.object(Uploader, 'upload_signal')
    def test_set_exception_true(self, mocked_upload_signal, mocked_nvml_read, mocked_host_read,
                            mocked_stop, mocked_start):
        trace = EndpointTrace(endpoint='ep1', profiler=TensorFlowProfiler())
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

    @patch.object(TensorFlowProfiler, 'start', return_value=True)
    @patch.object(TensorFlowProfiler, 'stop', return_value=True)
    @patch.object(ProcessReader, 'read')
    @patch.object(NvmlReader, 'read')
    @patch.object(Uploader, 'upload_signal')
    def test_set_data(self, mocked_upload_signal, mocked_nvml_read, mocked_host_read,
                            mocked_stop, mocked_start):
        
        for i in range(MissingValueDetector.MIN_BASELINE_SIZE):
            with EndpointTrace(endpoint='ep1', profiler=TensorFlowProfiler()) as trace:
                trace.set_data('d1', {'c1': i, 'c2': 1})
        with EndpointTrace(endpoint='ep1', profiler=TensorFlowProfiler()) as trace:
            trace.set_data('d1', {'c1': 100, 'c2': None})

        signal = mocked_upload_signal.call_args[0][0]

        self.assertEqual(signal.endpoint_name, 'ep1')
        self.assertTrue(signal.worker_id != '')
        self.assertTrue(signal.start_us > 0)
        self.assertTrue(signal.end_us > 0)
        self.assertEqual(signal.signal_type, signals_pb2.SignalType.MISSING_VALUES_SIGNAL)
