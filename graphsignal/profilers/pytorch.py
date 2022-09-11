from typing import Optional
import logging
import os
import sys
import time
import gzip
import torch
from torch.autograd import DeviceType
import torch.distributed

import graphsignal
from graphsignal.proto_utils import parse_semver
from graphsignal.proto import signals_pb2
from graphsignal.profilers.operation_profiler import OperationProfiler
from graphsignal.profilers.profiler_utils import create_log_dir, remove_log_dir

logger = logging.getLogger('graphsignal')


class PyTorchProfiler(OperationProfiler):
    def __init__(self):
        self._torch_prof = None
        self._log_dir = None
        self._pytorch_version = None

    def read_info(self, signal):
        if not self._pytorch_version:
            self._pytorch_version = signals_pb2.SemVer()
            parse_semver(self._pytorch_version, torch.__version__)

        # Framework info
        framework = signal.frameworks.add()
        framework.type = signals_pb2.FrameworkInfo.FrameworkType.PYTORCH_FRAMEWORK
        framework.version.CopyFrom(self._pytorch_version)

    def start(self, signal):
        logger.debug('Activating PyTorch profiler')

        if not self._torch_prof:
            # Initialization
            def _schedule_func(step):
                return torch.profiler.ProfilerAction.RECORD

            self._torch_prof = torch.profiler.profile(
                schedule=_schedule_func,
                record_shapes=False,
                profile_memory=True,
                with_stack=False,
                with_flops=True)

        # Profiler info
        signal.agent_info.operation_profiler_type = signals_pb2.AgentInfo.ProfilerType.PYTORCH_PROFILER

        self._torch_prof.start()

    def stop(self, signal):
        logger.debug('Deactivating PyTorch profiler')

        self._torch_prof.stop()

        self._convert_operations(signal)

        self._read_chrome_trace(signal)

    def _convert_operations(self, signal):
        # Operation stats
        for event_avg in self._torch_prof.key_averages():
            if event_avg.key and event_avg.key.startswith('ProfilerStep'):
                continue
            op_stats = signal.op_stats.add()
            op_stats.op_name = event_avg.key
            op_stats.count = _uint(event_avg.count)
            op_stats.total_host_time_us = _uint(event_avg.cpu_time_total)
            op_stats.total_device_time_us = _uint(event_avg.cuda_time_total)
            op_stats.self_host_time_us = _uint(event_avg.self_cpu_time_total)
            op_stats.self_device_time_us = _uint(event_avg.self_cuda_time_total)
            op_stats.total_host_memory = _uint(event_avg.cpu_memory_usage)
            op_stats.total_device_memory = _uint(event_avg.cuda_memory_usage)
            op_stats.self_host_memory = _uint(event_avg.self_cpu_memory_usage)
            op_stats.self_device_memory = _uint(event_avg.self_cuda_memory_usage)
            op_stats.flops = _uint(event_avg.flops)
            if event_avg.device_type in (DeviceType.CUDA, DeviceType.HIP)  or op_stats.self_device_time_us > 0:
                op_stats.device_type = signals_pb2.DeviceType.GPU
            else:
                op_stats.device_type = signals_pb2.DeviceType.CPU

        # Kernel stats
        kernel_index = {}
        for event in self._torch_prof.events():
            for kernel in event.kernels:
                key = (event.key, kernel.name, kernel.device)
                if key in kernel_index:
                    kernel_stats = kernel_index[key]
                    kernel_stats.count += 1
                    kernel_stats.duration_ns += _uint(kernel.duration * 1000)
                else:
                    kernel_stats = kernel_index[key] = signal.kernel_stats.add()
                    kernel_stats.device_type = signals_pb2.DeviceType.GPU
                    kernel_stats.device_id = str(kernel.device)
                    kernel_stats.op_name = event.name
                    kernel_stats.kernel_name = kernel.name
                    kernel_stats.count = 1
                    kernel_stats.duration_ns = _uint(kernel.duration * 1000)

        for kernel_stats in kernel_index.values():
            signal.kernel_stats.append(kernel_stats)

        logger.debug(
            'Converted %d PyTorch operation statistics', len(signal.op_stats))

    def _read_chrome_trace(self, signal):
        try:
            self._log_dir = create_log_dir()

            trace_path = os.path.join(self._log_dir, 'trace.json')
            self._torch_prof.export_chrome_trace(trace_path)

            trace_file_size = os.path.getsize(trace_path)
            if trace_file_size > 50 * 1e6:
                raise Exception('Trace file too big: {0}'.format(trace_file_size))

            with open(trace_path) as f:
                trace_json = f.read()
                signal.trace_data = gzip.compress(trace_json.encode())
        finally:
            remove_log_dir(self._log_dir)

def _uint(val):
    return max(int(val), 0)
