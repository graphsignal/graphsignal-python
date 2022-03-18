import logging
import os
import sys
import time
import tempfile
import gzip
import shutil
import glob
import tensorflow as tf
from tensorflow.python.client import device_lib
from google.protobuf.json_format import Parse

from graphsignal.profilers.tensorflow_proto import overview_page_pb2
from graphsignal.profilers.tensorflow_proto import input_pipeline_pb2
from graphsignal.profilers.tensorflow_proto import tf_stats_pb2
from graphsignal.profilers.tensorflow_proto import kernel_stats_pb2
from graphsignal.profilers.tensorflow_proto import memory_profile_pb2

import graphsignal
from graphsignal.system_info import parse_semver, compare_semver
from graphsignal.proto import profiles_pb2

logger = logging.getLogger('graphsignal')


class TensorflowProfiler():
    __slots__ = [
        '_is_initialized',
        '_log_dir',
        '_run_env'
    ]

    def __init__(self):
        self._is_initialized = False
        self._log_dir = None
        self._run_env = None

    def start(self):
        logger.debug('Activating TensorFlow profiler')

        if not self._is_initialized:
            logger.debug('Warming up TensorFlow profiler before first use')
            self._is_initialized = True
            tf.profiler.experimental.start('')
            tf.profiler.experimental.stop(save=False)
            logger.debug('Finished warming up')

            self._read_run_env()

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

        return True

    def stop(self, profile):
        logger.debug('Deactivating TensorFlow profiler')

        try:
            tf.profiler.experimental.stop()

            self._copy_run_env(profile)
            self._convert_to_profile(profile)
        except Exception as e:
            raise e
        finally:
            self._remove_log_dir()

        return True

    def _create_log_dir(self):
        self._log_dir = tempfile.mkdtemp(prefix='graphsignal-')
        logger.debug('Created temporary log directory %s', self._log_dir)

    def _remove_log_dir(self):
        shutil.rmtree(self._log_dir)
        logger.debug('Removed temporary log directory %s', self._log_dir)

    def _read_run_env(self):
        self._run_env = profiles_pb2.RunEnvironment()
        self._run_env.ml_framework = profiles_pb2.RunEnvironment.MLFramework.TENSORFLOW
        parse_semver(self._run_env.ml_framework_version, tf.__version__)
        if compare_semver(self._run_env.ml_framework_version, (2, 2, 0)) == -1:
            raise Exception(
                'TensorFlow profiling is not supported for versions <=2.2')
        for device in tf.config.list_physical_devices('GPU'):
            device_proto = self._run_env.devices.add()
            device_proto.type = profiles_pb2.DeviceType.GPU
            details = tf.config.experimental.get_device_details(device)
            if details:
                if 'device_name' in details:
                    device_proto.name = details['device_name']
                if 'compute_capability' in details:
                    device_proto.is_cuda_enabled = True
                    device_proto.compute_capability.major = details['compute_capability'][0]
                    device_proto.compute_capability.minor = details['compute_capability'][1]

    def _copy_run_env(self, profile):
        profile.run_env.ml_framework = self._run_env.ml_framework
        profile.run_env.ml_framework_version.CopyFrom(
            self._run_env.ml_framework_version)
        profile.run_env.devices.extend(self._run_env.devices)

    def _convert_to_profile(self, profile):
        overview_page_data = self._find_and_read(
            'plugins/profile/*/*overview_page.pb')
        if overview_page_data:
            overview_page = overview_page_pb2.OverviewPage()
            overview_page.ParseFromString(overview_page_data)
            analysis = overview_page.analysis
            profile.summary.device_idle_percent = analysis.device_idle_time_percent
            profile.summary.host_idle_percent = analysis.host_idle_time_percent
            profile.summary.device_compute_16bit_percent = analysis.device_compute_16bit_percent
            profile.summary.device_compute_32bit_percent = analysis.device_compute_32bit_percent
            profile.summary.host_op_percent = analysis.host_tf_op_percent
            profile.summary.device_op_percent = analysis.device_tf_op_percent
            profile.summary.mxu_utilization = analysis.mxu_utilization_percent

        tf_stats_data = self._find_and_read(
            'plugins/profile/*/*tensorflow_stats.pb')
        if tf_stats_data:
            tf_stats_db = tf_stats_pb2.TfStatsDatabase()
            tf_stats_db.ParseFromString(tf_stats_data)
            for tf_stats_record in tf_stats_db.without_idle.tf_stats_record:
                op_stats = profile.op_stats.add()
                if tf_stats_record.host_or_device == 'Host':
                    op_stats.device_type = profiles_pb2.DeviceType.CPU
                    op_stats.total_host_time_us = int(
                        tf_stats_record.total_time_in_us)
                    op_stats.self_host_time_us = int(
                        tf_stats_record.total_self_time_in_us)
                    op_stats.self_host_memory_rate = int(
                        tf_stats_record.measured_memory_bw)
                else:
                    op_stats.device_type = profiles_pb2.DeviceType.GPU
                    op_stats.total_device_time_us = int(
                        tf_stats_record.total_time_in_us)
                    op_stats.self_device_time_us = int(
                        tf_stats_record.total_self_time_in_us)
                    op_stats.self_device_memory_rate = int(
                        tf_stats_record.measured_memory_bw)
                    op_stats.tensorcore_utilization = tf_stats_record.gpu_tensorcore_utilization
                op_stats.op_type = tf_stats_record.op_type
                op_stats.op_name = tf_stats_record.op_name
                op_stats.count = int(tf_stats_record.occurrences)
        else:
            logger.debug('No operation data found in TensorFlow log directory')

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
                kernel_stats.count = int(kernel_report.occurrences)
                kernel_stats.duration_ns = int(kernel_report.total_duration_ns)
                kernel_stats.is_using_tensorcore = kernel_report.is_kernel_using_tensor_core
        else:
            logger.debug('No kernel data found in TensorFlow log directory')

    def _find_and_read(self, file_pattern):
        file_paths = glob.glob(os.path.join(self._log_dir, file_pattern))
        if len(file_paths) == 0:
            raise Exception(
                'Files are not found at {}'.format(
                    os.path.join(
                        self._log_dir,
                        file_pattern)))

        if file_paths[-1].endswith('.gz'):
            last_file = gzip.open(file_paths[-1], "rb")
        else:
            last_file = open(file_paths[-1], "rb")
        data = last_file.read()
        last_file.close()

        return data
