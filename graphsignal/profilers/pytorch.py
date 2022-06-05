from typing import Optional
import logging
import os
import sys
import time
import tempfile
import gzip
import shutil
import glob
import torch
from torch.autograd import DeviceType
import torch.distributed

import graphsignal
from graphsignal.proto_utils import parse_semver
from graphsignal.proto import profiles_pb2
from graphsignal.profiling_step import ProfilingStep
from graphsignal.profilers.framework_profiler import FrameworkProfiler

logger = logging.getLogger('graphsignal')


class PyTorchProfiler(FrameworkProfiler):
    def __init__(self):
        self._torch_prof = None
        self._log_dir = None
        self._ml_framework = None
        self._ml_framework_version = None
        self._global_rank = None
        self._world_size = None
        self._comm_backend = None

    def start(self, profile):
        logger.debug('Activating PyTorch profiler')

        if not self._torch_prof:
            # Initialization
            logger.debug('Warming up PyTorch profiler before first use')

            def _schedule_func(step):
                return torch.profiler.ProfilerAction.RECORD

            self._torch_prof = torch.profiler.profile(
                schedule=_schedule_func,
                record_shapes=False,
                profile_memory=True,
                with_stack=False,
                with_flops=True)
            self._torch_prof.start()
            self._torch_prof.stop()
            logger.debug('Finished warming up')

            self._ml_framework = profiles_pb2.ProcessUsage.MLFramework.Value('PYTORCH')
            self._ml_framework_version = profiles_pb2.SemVer()
            parse_semver(self._ml_framework_version, torch.__version__)

            if torch.distributed.is_available():
                if torch.distributed.is_initialized():
                    self._global_rank = torch.distributed.get_rank()
                    self._world_size = torch.distributed.get_world_size()
                    self._comm_backend = torch.distributed.get_backend()

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

        # Communication info
        if self._comm_backend:
            if self._comm_backend == 'nccl':
                profile.comm_usage.backend_type = profiles_pb2.CommunicationUsage.CommunicationBackendType.NCCL
            if self._comm_backend == 'gloo':
                profile.comm_usage.backend_type = profiles_pb2.CommunicationUsage.CommunicationBackendType.GLOO
            if self._comm_backend == 'mpi':
                profile.comm_usage.backend_type = profiles_pb2.CommunicationUsage.CommunicationBackendType.MPI

        self._torch_prof.start()

    def stop(self, profile):
        logger.debug('Deactivating PyTorch profiler')

        self._torch_prof.stop()

        self._convert_operations(profile)

        # Chrome trace
        try:
            self._create_log_dir()
            trace_path = os.path.join(self._log_dir, 'trace.json')
            self._torch_prof.export_chrome_trace(trace_path)
            with open(trace_path) as f:
                trace_json = f.read()
                profile.trace_data = gzip.compress(trace_json.encode())
        except Exception as e:
            logger.error('Error exporting Chrome trace', exc_info=True)
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
        for event_avg in self._torch_prof.key_averages():
            if event_avg.key and event_avg.key.startswith('ProfilerStep'):
                continue
            op_stats = profile.op_stats.add()
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
                op_stats.device_type = profiles_pb2.DeviceType.GPU
            else:
                op_stats.device_type = profiles_pb2.DeviceType.CPU

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
                    kernel_stats = kernel_index[key] = profile.kernel_stats.add()
                    kernel_stats.device_type = profiles_pb2.DeviceType.GPU
                    kernel_stats.device_id = str(kernel.device)
                    kernel_stats.op_name = event.name
                    kernel_stats.kernel_name = kernel.name
                    kernel_stats.count = 1
                    kernel_stats.duration_ns = _uint(kernel.duration * 1000)

        for kernel_stats in kernel_index.values():
            profile.kernel_stats.append(kernel_stats)

        logger.debug(
            'Converted %d PyTorch operation statistics', len(profile.op_stats))

def _uint(val):
    return max(int(val), 0)


_profiler = PyTorchProfiler()

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
