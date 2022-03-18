import logging
import os
import sys
import time
import tempfile
import shutil
import glob
import torch
from torch.autograd import DeviceType

import graphsignal
from graphsignal.system_info import parse_semver
from graphsignal.proto import profiles_pb2

logger = logging.getLogger('graphsignal')


class PytorchProfiler():
    __slots__ = [
        '_torch_prof',
        '_run_env'
    ]

    def __init__(self):
        self._torch_prof = None
        self._run_env = None

    def start(self):
        logger.debug('Activating PyTorch profiler')

        if not self._torch_prof:
            logger.debug('Warming up PyTorch profiler before first use')

            def _schedule_func(step):
                return torch.profiler.ProfilerAction.RECORD

            self._torch_prof = torch.profiler.profile(
                schedule=_schedule_func,
                record_shapes=False,
                with_stack=False,
                with_flops=True)
            self._torch_prof.start()
            self._torch_prof.stop()
            logger.debug('Finished warming up')

            self._read_run_env()

        self._torch_prof.start()

        return True

    def stop(self, profile):
        logger.debug('Deactivating PyTorch profiler')

        self._torch_prof.stop()

        self._copy_run_env(profile)
        self._convert_to_profile(profile)

        return True

    def _read_run_env(self):
        self._run_env = profiles_pb2.RunEnvironment()
        self._run_env.ml_framework = profiles_pb2.RunEnvironment.MLFramework.Value('PYTORCH')
        parse_semver(self._run_env.ml_framework_version, torch.__version__)
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                device_proto = self._run_env.devices.add()
                device_proto.type = profiles_pb2.DeviceType.GPU
                device_proto.name = torch.cuda.get_device_name(index)
                device_proto.is_cuda_enabled = True

                device_properties = torch.cuda.get_device_properties(index)
                device_proto.compute_capability.major = device_properties.major
                device_proto.compute_capability.minor = device_properties.minor
                device_proto.total_memory = device_properties.total_memory

    def _copy_run_env(self, profile):
        profile.run_env.ml_framework = self._run_env.ml_framework
        profile.run_env.ml_framework_version.CopyFrom(
            self._run_env.ml_framework_version)
        profile.run_env.devices.extend(self._run_env.devices)

    def _convert_to_profile(self, profile):
        for event_avg in self._torch_prof.key_averages():
            op_stats = profile.op_stats.add()
            if event_avg.device_type == DeviceType.CUDA:
                op_stats.device_type = profiles_pb2.DeviceType.GPU
            else:
                op_stats.device_type = profiles_pb2.DeviceType.CPU
            op_stats.op_name = event_avg.key
            op_stats.count = int(event_avg.count)
            op_stats.total_host_time_us = int(event_avg.cpu_time_total)
            op_stats.total_device_time_us = int(event_avg.cuda_time_total)
            op_stats.self_host_time_us = int(event_avg.self_cpu_time_total)
            op_stats.self_device_time_us = int(event_avg.self_cuda_time_total)
            op_stats.total_host_memory = int(event_avg.cpu_memory_usage)
            op_stats.total_device_memory = int(event_avg.cuda_memory_usage)
            op_stats.self_host_memory = int(event_avg.self_cpu_memory_usage)
            op_stats.self_device_memory = int(event_avg.self_cuda_memory_usage)
            op_stats.flops = int(event_avg.flops)

        kernel_index = {}
        for event in self._torch_prof.events():
            for kernel in event.kernels:
                key = (event.key, kernel.name, kernel.device)
                if key in kernel_index:
                    kernel_stats = kernel_index[key]
                    kernel_stats.count += 1
                    kernel_stats.duration_ns += int(kernel.duration * 1000)
                else:
                    kernel_stats = kernel_index[key] = profile.kernel_stats.add()
                    kernel_stats.device_type = profiles_pb2.DeviceType.GPU
                    kernel_stats.device_id = str(kernel.device)
                    kernel_stats.op_name = event.name
                    kernel_stats.kernel_name = kernel.name
                    kernel_stats.count = 1
                    kernel_stats.duration_ns = int(kernel.duration * 1000)

        for kernel_stats in kernel_index.values():
            profile.kernel_stats.append(kernel_stats)

        sum_host_op_time_us = 0
        sum_device_op_time_us = 0
        for op_stats in profile.op_stats:
            sum_host_op_time_us += op_stats.self_host_time_us
            sum_device_op_time_us += op_stats.self_device_time_us
        sum_op_time_us = sum_host_op_time_us + sum_device_op_time_us
        if sum_op_time_us > 0:
            profile.summary.host_op_percent = sum_host_op_time_us / sum_op_time_us * 100
            profile.summary.device_op_percent = sum_device_op_time_us / sum_op_time_us * 100

        logger.debug(
            'Converted %d PyTorch operation statistics', len(profile.op_stats))
