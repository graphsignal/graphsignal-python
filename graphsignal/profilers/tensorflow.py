from typing import Optional
import logging
import os
import sys
import time
import tempfile
import gzip
import shutil
import glob
import json
import tensorflow as tf
from tensorflow.python.client import device_lib
from google.protobuf.json_format import Parse

from graphsignal.profilers.tensorflow_proto import overview_page_pb2
from graphsignal.profilers.tensorflow_proto import input_pipeline_pb2
from graphsignal.profilers.tensorflow_proto import tf_stats_pb2
from graphsignal.profilers.tensorflow_proto import kernel_stats_pb2
from graphsignal.profilers.tensorflow_proto import memory_profile_pb2

import graphsignal
from graphsignal.proto_utils import parse_semver, compare_semver
from graphsignal.proto import profiles_pb2
from graphsignal.profiling_step import ProfilingStep
from graphsignal.profilers.framework_profiler import FrameworkProfiler

logger = logging.getLogger('graphsignal')


class TensorflowProfiler(FrameworkProfiler):
    def __init__(self):
        self._is_initialized = False
        self._log_dir = None
        self._ml_framework = None
        self._ml_framework_version = None
        self._global_rank = None
        self._world_size = None

    def start(self, profile):
        logger.debug('Activating TensorFlow profiler')

        # Initialization
        if not self._is_initialized:
            logger.debug('Warming up TensorFlow profiler before first use')
            self._is_initialized = True
            tf.profiler.experimental.start('')
            tf.profiler.experimental.stop(save=False)
            logger.debug('Finished warming up')

            self._ml_framework = profiles_pb2.ProcessUsage.MLFramework.TENSORFLOW
            self._ml_framework_version = profiles_pb2.SemVer()
            parse_semver(self._ml_framework_version, tf.__version__)
            if compare_semver(self._ml_framework_version, (2, 2, 0)) == -1:
                raise Exception(
                    'TensorFlow profiling is not supported for versions <=2.2')

            if 'TF_CONFIG' in os.environ:
                try:
                    tf_config = json.loads(os.environ['TF_CONFIG'])
                    self._world_size = 0
                    if 'chief' in tf_config['cluster']:
                        self._world_size += len(tf_config['cluster']['chief'])
                    if 'worker' in tf_config['cluster']:
                        self._world_size += len(tf_config['cluster']['worker'])
                    self._global_rank = tf_config['task']['index']
                except:
                    logger.warning('Error parsing TF_CONFIG', exc_info=True)

        # Process info
        profile.process_usage.ml_framework = self._ml_framework
        profile.process_usage.ml_framework_version.CopyFrom(
            self._ml_framework_version)
        if self._global_rank is not None and self._global_rank >= 0:
            if graphsignal._agent.global_rank == -1:
                profile.process_usage.global_rank = self._global_rank

        # Step stats
        if self._world_size is not None and self._world_size > 0:
            profile.step_stats.world_size = self._world_size
            graphsignal.log_parameter('world_size', self._world_size)

        try:
            self._create_log_dir()

            options = tf.profiler.experimental.ProfilerOptions(
                host_tracer_level=2,
                python_tracer_level=1,
                device_tracer_level=1)
            tf.profiler.experimental.start(self._log_dir, options=options)
        except Exception as e:
            self._remove_log_dir()
            raise e

    def stop(self, profile):
        logger.debug('Deactivating TensorFlow profiler')

        try:
            tf.profiler.experimental.stop()

            self._convert_operations(profile)

            trace_json_gz = self._find_and_read(
                'plugins/profile/*/*trace.json.gz',
                decompress=False)
            profile.trace_data = trace_json_gz
        except Exception as e:
            raise e
        finally:
            self._remove_log_dir()

    def _create_log_dir(self):
        self._log_dir = tempfile.mkdtemp(prefix='graphsignal-')
        logger.debug('Created temporary log directory %s', self._log_dir)

    def _remove_log_dir(self):
        shutil.rmtree(self._log_dir)
        logger.debug('Removed temporary log directory %s', self._log_dir)

    def _convert_operations(self, profile):
        # Operation stats
        tf_stats_data = self._find_and_read(
            'plugins/profile/*/*tensorflow_stats.pb')
        if tf_stats_data:
            tf_stats_db = tf_stats_pb2.TfStatsDatabase()
            tf_stats_db.ParseFromString(tf_stats_data)
            for tf_stats_record in tf_stats_db.without_idle.tf_stats_record:
                op_stats = profile.op_stats.add()
                if tf_stats_record.host_or_device == 'Host':
                    op_stats.device_type = profiles_pb2.DeviceType.CPU
                    op_stats.total_host_time_us = _uint(tf_stats_record.total_time_in_us)
                    op_stats.self_host_time_us = _uint(tf_stats_record.total_self_time_in_us)
                    op_stats.self_host_memory_rate = _uint(tf_stats_record.measured_memory_bw)
                else:
                    op_stats.device_type = profiles_pb2.DeviceType.GPU
                    op_stats.total_device_time_us = _uint(tf_stats_record.total_time_in_us)
                    op_stats.self_device_time_us = _uint(tf_stats_record.total_self_time_in_us)
                    op_stats.self_device_memory_rate = _uint(tf_stats_record.measured_memory_bw)
                    op_stats.tensorcore_utilization = tf_stats_record.gpu_tensorcore_utilization
                op_stats.op_type = tf_stats_record.op_type
                op_stats.op_name = tf_stats_record.op_name
                op_stats.count = _uint(tf_stats_record.occurrences)
                op_stats.flops_per_sec = _uint(tf_stats_record.measured_flop_rate)
        else:
            logger.debug('No operation data found in TensorFlow log directory')

        # Kernel stats
        kernel_stats_data = self._find_and_read(
            'plugins/profile/*/*kernel_stats.pb')
        if kernel_stats_data:
            kernel_stats_db = kernel_stats_pb2.KernelStatsDb()
            kernel_stats_db.ParseFromString(kernel_stats_data)
            for kernel_report in kernel_stats_db.reports:
                kernel_stats = profile.kernel_stats.add()
                kernel_stats.device_type = profiles_pb2.DeviceType.GPU
                kernel_stats.op_name = kernel_report.op_name
                kernel_stats.kernel_name = kernel_report.name
                kernel_stats.count = _uint(kernel_report.occurrences)
                kernel_stats.duration_ns = _uint(kernel_report.total_duration_ns)
                kernel_stats.is_using_tensorcore = kernel_report.is_kernel_using_tensor_core
        else:
            logger.debug('No kernel data found in TensorFlow log directory')

    def _find_and_read(self, file_pattern, decompress=True):
        file_paths = glob.glob(os.path.join(self._log_dir, file_pattern))
        if len(file_paths) == 0:
            raise Exception(
                'Files are not found at {}'.format(
                    os.path.join(
                        self._log_dir,
                        file_pattern)))

        if decompress and file_paths[-1].endswith('.gz'):
            last_file = gzip.open(file_paths[-1], "rb")
        else:
            last_file = open(file_paths[-1], "rb")
        data = last_file.read()
        last_file.close()

        return data


def _uint(val):
    return max(int(val), 0)


_profiler = TensorflowProfiler()

def profile_step(
        phase_name: Optional[str] = None,
        effective_batch_size: Optional[int] = None,
        ensure_profile: Optional[bool] = False) -> ProfilingStep:
    graphsignal._check_configured()

    return ProfilingStep(
        phase_name=phase_name,
        effective_batch_size=effective_batch_size,
        ensure_profile=ensure_profile,
        framework_profiler=_profiler)
