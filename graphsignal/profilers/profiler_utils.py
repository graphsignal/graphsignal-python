import logging
import os
import tempfile
import gzip
import shutil
import glob

from graphsignal.profilers.tensorflow_proto import overview_page_pb2
from graphsignal.profilers.tensorflow_proto import input_pipeline_pb2
from graphsignal.profilers.tensorflow_proto import tf_stats_pb2
from graphsignal.profilers.tensorflow_proto import kernel_stats_pb2
from graphsignal.profilers.tensorflow_proto import memory_profile_pb2

from graphsignal.proto import profiles_pb2

logger = logging.getLogger('graphsignal')

def create_log_dir():
    log_dir = tempfile.mkdtemp(prefix='graphsignal-')
    logger.debug('Created temporary log directory %s', log_dir)
    return log_dir

def remove_log_dir(log_dir):
    shutil.rmtree(log_dir)
    logger.debug('Removed temporary log directory %s', log_dir)

def find_and_read(log_dir, file_pattern, decompress=True, max_size=None):
    file_paths = glob.glob(os.path.join(log_dir, file_pattern))
    if len(file_paths) == 0:
        raise Exception(
            'Files are not found at {}'.format(
                os.path.join(
                    log_dir,
                    file_pattern)))

    found_path = file_paths[-1]

    if max_size:
        if os.path.getsize(found_path) > max_size:
            raise Exception('File is too big: {0}'.format())

    if decompress and found_path.endswith('.gz'):
        last_file = gzip.open(found_path, "rb")
    else:
        last_file = open(found_path, "rb")
    data = last_file.read()
    last_file.close()

    return data

def convert_tensorflow_profile(log_dir, profile):
    # Read trace
    trace_json_gz = find_and_read(
        log_dir,
        'plugins/profile/*/*trace.json.gz',
        decompress=False,
        max_size=5 * 1e6)
    if trace_json_gz:
        profile.trace_data = trace_json_gz

    # Operation stats
    tf_stats_data = find_and_read(
        log_dir,
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
        logger.debug('No operation data found in log directory')

    # Kernel stats
    kernel_stats_data = find_and_read(
        log_dir,
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
        logger.debug('No kernel data found in log directory')

def _uint(val):
    return max(int(val), 0)
