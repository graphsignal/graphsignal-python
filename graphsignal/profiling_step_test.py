import unittest
import logging
import sys
import time
from unittest.mock import patch, Mock

import graphsignal
from graphsignal.proto import profiles_pb2
from graphsignal.profiling_step import ProfilingStep
from graphsignal.profilers.tensorflow import TensorflowProfiler
from graphsignal.usage.process_reader import ProcessReader
from graphsignal.usage.nvml_reader import NvmlReader
from graphsignal.uploader import Uploader

logger = logging.getLogger('graphsignal')


class ProfilingStepTest(unittest.TestCase):
    def setUp(self):
        if len(logger.handlers) == 0:
            logger.addHandler(logging.StreamHandler(sys.stdout))
        graphsignal.configure(
            api_key='k1',
            workload_name='w1',
            run_id='r1',
            local_rank=1,
            global_rank=1,
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
        graphsignal.add_tag('t1')
        graphsignal.add_tag('t1')
        graphsignal.add_tag('t2')

        graphsignal.log_parameter('n1', 'v1')
        graphsignal.log_parameter('n1', 'v2')
        graphsignal.log_parameter('n3', 'v3')

        graphsignal.log_metric('m1', 1)
        graphsignal.log_metric('m1', 2.2)
        graphsignal.log_metric('m3', 3)

        step = ProfilingStep(
            phase_name='training',
            effective_batch_size=128,
            ensure_profile=True,
            framework_profiler=TensorflowProfiler())
        step.set_effective_batch_size(256)
        step.stop()

        mocked_start.assert_called_once()
        mocked_stop.assert_called_once()
        profile = mocked_upload_profile.call_args[0][0]

        self.assertEqual(profile.workload_name, 'w1')
        self.assertTrue(profile.worker_id != '')
        self.assertEqual(profile.run_id, '5573e39b6600')
        self.assertEqual(profile.phase_name, 'training')
        self.assertTrue(profile.start_us > 0)
        self.assertTrue(profile.end_us > 0)
        self.assertEqual(profile.step_stats.step_count, 1)
        self.assertTrue(profile.step_stats.total_time_us >= 0)
        self.assertEqual(profile.step_stats.sample_count, 256)
        self.assertEqual(len(profile.tags), 2)
        self.assertEqual(profile.tags[0].value, 't1')
        self.assertEqual(profile.tags[1].value, 't2')
        self.assertEqual(profile.params[0].name, 'n1')
        self.assertEqual(profile.params[0].value, 'v2')
        self.assertEqual(profile.params[1].name, 'n3')
        self.assertEqual(profile.params[1].value, 'v3')
        self.assertEqual(profile.metrics[0].name, 'm1')
        self.assertEqual(profile.metrics[0].value, 2.2)
        self.assertEqual(profile.metrics[1].name, 'm3')
        self.assertEqual(profile.metrics[1].value, 3)
        self.assertEqual(profile.node_usage.node_rank, -1)
        self.assertTrue(profile.process_usage.start_ms > 0)
        self.assertEqual(profile.process_usage.local_rank, 1)
        self.assertEqual(profile.process_usage.global_rank, 1)

    @patch.object(TensorflowProfiler, 'start', return_value=True)
    @patch.object(TensorflowProfiler, 'stop', return_value=True)
    @patch.object(ProcessReader, 'read')
    @patch.object(NvmlReader, 'read')
    @patch.object(Uploader, 'upload_profile')
    def test_start_exception(self, mocked_upload_profile, mocked_nvml_read, mocked_host_read,
                             mocked_stop, mocked_start):
        mocked_start.side_effect = Exception('ex1')
        step = ProfilingStep(
            ensure_profile=True,
            framework_profiler=TensorflowProfiler())
        step.stop()

        mocked_start.assert_called_once()
        mocked_stop.assert_not_called()
        profile = mocked_upload_profile.call_args[0][0]

        self.assertEqual(profile.workload_name, 'w1')
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
    def test_stop_exception(self, mocked_upload_profile, mocked_nvml_read, mocked_host_read,
                            mocked_stop, mocked_start):
        mocked_stop.side_effect = Exception('ex1')
        step = ProfilingStep(
            ensure_profile=True,
            framework_profiler=TensorflowProfiler())
        step.stop()

        mocked_start.assert_called_once()
        mocked_stop.assert_called_once()
        profile = mocked_upload_profile.call_args[0][0]

        self.assertEqual(profile.workload_name, 'w1')
        self.assertTrue(profile.worker_id != '')
        self.assertTrue(profile.start_us > 0)
        self.assertTrue(profile.end_us > 0)
        self.assertEqual(profile.profiler_errors[0].message, 'ex1')
        self.assertNotEqual(profile.profiler_errors[0].stack_trace, '')
