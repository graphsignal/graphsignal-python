import unittest
import logging
import sys
import time
from unittest.mock import patch, Mock

import graphsignal
from graphsignal.proto import profiles_pb2
from graphsignal.profiling_span import ProfilingSpan
from graphsignal.profilers.tensorflow import TensorflowProfiler
from graphsignal.usage.process_reader import ProcessReader
from graphsignal.usage.nvml_reader import NvmlReader
from graphsignal.uploader import Uploader

logger = logging.getLogger('graphsignal')


class ProflingSpanTest(unittest.TestCase):
    def setUp(self):
        if len(logger.handlers) == 0:
            logger.addHandler(logging.StreamHandler(sys.stdout))
        graphsignal.configure(
            api_key='k1',
            workload_name='w1',
            debug_mode=True)

    def tearDown(self):
        graphsignal.shutdown()

    @patch.object(TensorflowProfiler, 'start', return_value=True)
    @patch.object(TensorflowProfiler, 'stop', return_value=True)
    @patch.object(ProcessReader, 'read')
    @patch.object(NvmlReader, 'read')
    @patch.object(Uploader, 'upload_profile')
    def test_start_stop(self, mocked_upload_profile, mocked_nvml_read, mocked_host_read,
                        mocked_stop, mocked_start):
        span = ProfilingSpan(
            framework_profiler=TensorflowProfiler(),
            span_name='s1',
            span_type=profiles_pb2.Span.SpanType.TRAINING_BATCH,
            ensure_profile=True)
        span.add_metadata('m1', 'v1')
        span.add_metadata('m2', 'v2')
        span.stop()

        mocked_start.assert_called_once()
        mocked_stop.assert_called_once()
        profile = mocked_upload_profile.call_args[0][0]

        self.assertEqual(profile.workload_name, 'w1')
        self.assertTrue(profile.run_id != '')
        self.assertEqual(profile.span.name, 's1')
        self.assertEqual(profile.span.type, profiles_pb2.Span.SpanType.TRAINING_BATCH)
        self.assertTrue(profile.start_us > 0)
        self.assertTrue(profile.end_us > 0)
        self.assertEqual(profile.metadata[0].key, 'm1')
        self.assertEqual(profile.metadata[0].value, 'v1')
        self.assertEqual(profile.metadata[1].key, 'm2')
        self.assertEqual(profile.metadata[1].value, 'v2')

    @patch.object(TensorflowProfiler, 'start', return_value=True)
    @patch.object(TensorflowProfiler, 'stop', return_value=True)
    @patch.object(ProcessReader, 'read')
    @patch.object(NvmlReader, 'read')
    @patch.object(Uploader, 'upload_profile')
    def test_start_exception(self, mocked_upload_profile, mocked_nvml_read, mocked_host_read,
                             mocked_stop, mocked_start):
        mocked_start.side_effect = Exception('ex1')
        span = ProfilingSpan(
            framework_profiler=TensorflowProfiler(),
            span_name='s1',
            ensure_profile=True)
        span.stop()

        mocked_start.assert_called_once()
        mocked_stop.assert_not_called()
        profile = mocked_upload_profile.call_args[0][0]

        self.assertEqual(profile.workload_name, 'w1')
        self.assertTrue(profile.run_id != '')
        self.assertTrue(profile.start_us > 0)
        self.assertTrue(profile.end_us > 0)
        self.assertEqual(profile.span.name, 's1')
        self.assertTrue(profile.span.duration_us >= 0)
        self.assertEqual(profile.profiler_errors[0].message, 'ex1')
        self.assertNotEqual(profile.profiler_errors[0].stack_trace, '')

    @patch.object(TensorflowProfiler, 'start', return_value=True)
    @patch.object(TensorflowProfiler, 'stop', return_value=True)
    @patch.object(ProcessReader, 'read')
    @patch.object(NvmlReader, 'read')
    @patch.object(Uploader, 'upload_profile')
    def test_stop_exception(self, mocked_upload_profile, mocked_nvml_read, mocked_host_read,
                            mocked_stop, mocked_start):
        mocked_stop.side_effect = Exception('ex1')
        span = ProfilingSpan(
            framework_profiler=TensorflowProfiler(),
            span_name='s1',
            ensure_profile=True)
        span.stop()

        mocked_start.assert_called_once()
        mocked_stop.assert_called_once()
        profile = mocked_upload_profile.call_args[0][0]

        self.assertEqual(profile.workload_name, 'w1')
        self.assertTrue(profile.run_id != '')
        self.assertEqual(profile.span.name, 's1')
        self.assertTrue(profile.start_us > 0)
        self.assertTrue(profile.end_us > 0)
        self.assertTrue(profile.span.duration_us >= 0)
        self.assertEqual(profile.profiler_errors[0].message, 'ex1')
        self.assertNotEqual(profile.profiler_errors[0].stack_trace, '')
