import unittest
import logging
import sys
import time
from unittest.mock import patch, Mock

import graphsignal
from graphsignal.proto import profiles_pb2
from graphsignal.inference_span import InferenceSpan
from graphsignal.tracers.tensorflow import TensorflowProfiler
from graphsignal.usage.process_reader import ProcessReader
from graphsignal.usage.nvml_reader import NvmlReader
from graphsignal.uploader import Uploader

logger = logging.getLogger('graphsignal')


class InferenceSpanTest(unittest.TestCase):
    def setUp(self):
        if len(logger.handlers) == 0:
            logger.addHandler(logging.StreamHandler(sys.stdout))
        graphsignal.configure(
            api_key='k1',
            workload_name='r1',
            debug_mode=True)

    def tearDown(self):
        graphsignal.shutdown()

    @patch.object(TensorflowProfiler, 'start', return_value=True)
    @patch.object(TensorflowProfiler, 'stop', return_value=True)
    @patch.object(ProcessReader, 'read')
    @patch.object(NvmlReader, 'read')
    @patch.object(Uploader, 'upload_profile')
    @patch('time.time', return_value=1000)
    def test_start_stop(self, mocked_time, mocked_upload_profile, mocked_nvml_read, mocked_host_read,
                        mocked_stop, mocked_start):
        for i in range(10):
            span = InferenceSpan(
                model_name='m1',
                metadata={'k1': 'v2', 'k3': 3.0},
                operation_profiler=TensorflowProfiler())
            span.set_count('items', 256)
            time.sleep(0.01)
            span.stop()

        self.assertEqual(mocked_start.call_count, 2)
        self.assertEqual(mocked_stop.call_count, 2)
        profile = mocked_upload_profile.call_args[0][0]

        self.assertEqual(profile.model_name, 'm1')
        self.assertTrue(profile.worker_id != '')
        self.assertEqual(profile.workload_id, '5573e39b6600')
        self.assertTrue(profile.start_us > 0)
        self.assertTrue(profile.end_us > 0)
        self.assertEqual(len(profile.inference_stats.inference_counter.buckets_sec), 1)
        self.assertEqual(len(profile.inference_stats.extra_counters['items'].buckets_sec), 1)
        self.assertEqual(len(profile.inference_stats.time_reservoir_us), 8)
        self.assertEqual(profile.metadata[0].key, 'k1')
        self.assertEqual(profile.metadata[0].value, 'v2')
        self.assertEqual(profile.metadata[1].key, 'k3')
        self.assertEqual(profile.metadata[1].value, '3.0')

    @patch.object(TensorflowProfiler, 'start', return_value=True)
    @patch.object(TensorflowProfiler, 'stop', return_value=True)
    @patch.object(ProcessReader, 'read')
    @patch.object(NvmlReader, 'read')
    @patch.object(Uploader, 'upload_profile')
    def test_start_exception(self, mocked_upload_profile, mocked_nvml_read, mocked_host_read,
                             mocked_stop, mocked_start):
        mocked_start.side_effect = Exception('ex1')
        span = InferenceSpan(
            model_name='m1',
            ensure_profile=True,
            operation_profiler=TensorflowProfiler())
        span.stop()

        mocked_start.assert_called_once()
        mocked_stop.assert_not_called()
        profile = mocked_upload_profile.call_args[0][0]

        self.assertEqual(profile.model_name, 'm1')
        self.assertTrue(profile.worker_id != '')
        self.assertTrue(profile.start_us > 0)
        self.assertTrue(profile.end_us > 0)
        self.assertEqual(profile.profiler_errors[0].message, 'ex1')
        self.assertNotEqual(profile.profiler_errors[0].stack_trace, '')

    @patch.object(TensorflowProfiler, 'start', return_value=True)
    @patch.object(TensorflowProfiler, 'stop', return_value=True)
    @patch.object(ProcessReader, 'read')
    @patch.object(NvmlReader, 'read')
    @patch.object(Uploader, 'upload_profile')
    def test_profiler_exception(self, mocked_upload_profile, mocked_nvml_read, mocked_host_read,
                            mocked_stop, mocked_start):
        mocked_stop.side_effect = Exception('ex1')
        span = InferenceSpan(
            model_name='m1',
            ensure_profile=True,
            operation_profiler=TensorflowProfiler())
        span.stop()

        mocked_start.assert_called_once()
        mocked_stop.assert_called_once()
        profile = mocked_upload_profile.call_args[0][0]

        self.assertEqual(profile.model_name, 'm1')
        self.assertTrue(profile.worker_id != '')
        self.assertTrue(profile.start_us > 0)
        self.assertTrue(profile.end_us > 0)
        self.assertEqual(profile.profiler_errors[0].message, 'ex1')
        self.assertNotEqual(profile.profiler_errors[0].stack_trace, '')

    @patch.object(TensorflowProfiler, 'start', return_value=True)
    @patch.object(TensorflowProfiler, 'stop', return_value=True)
    @patch.object(ProcessReader, 'read')
    @patch.object(NvmlReader, 'read')
    @patch.object(Uploader, 'upload_profile')
    def test_inference_exception(self, mocked_upload_profile, mocked_nvml_read, mocked_host_read,
                            mocked_stop, mocked_start):
        try:
            with InferenceSpan(model_name='m1', ensure_profile=True, operation_profiler=TensorflowProfiler()):
                raise Exception('ex1')
        except:
            pass

        mocked_start.assert_called_once()
        mocked_stop.assert_called_once()
        profile = mocked_upload_profile.call_args[0][0]

        self.assertEqual(profile.model_name, 'm1')
        self.assertTrue(profile.worker_id != '')
        self.assertTrue(profile.start_us > 0)
        self.assertTrue(profile.end_us > 0)
        self.assertEqual(profile.exceptions[0].message, 'ex1')
        self.assertNotEqual(profile.exceptions[0].stack_trace, '')
        self.assertEqual(len(profile.inference_stats.exception_counter.buckets_sec), 1)
