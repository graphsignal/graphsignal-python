import logging
import sys
import torch
import threading

import graphsignal
from graphsignal.recorders.base_recorder import BaseRecorder
from graphsignal.proto_utils import parse_semver
from graphsignal.proto import signals_pb2
from graphsignal.proto_utils import add_library_param, add_driver

logger = logging.getLogger('graphsignal')

class KinetoRecorder(BaseRecorder):
    def __init__(self):
        self._torch_prof = None
        self._profiler_lock = threading.Lock()

    def on_span_start(self, proto, context, options):
        if not options.enable_profiling:
            return
        if not self._profiler_lock.acquire(blocking=False):
            return

        if not self._torch_prof:
            def _schedule_func(step):
                return torch.profiler.ProfilerAction.RECORD
            self._torch_prof = torch.profiler.profile(
                schedule=_schedule_func,
                record_shapes=False,
                profile_memory=True,
                with_stack=False,
                with_flops=True)

        self._torch_prof.start()
        context['is_kineto_profiling'] = True

    def on_span_stop(self, proto, context, options):
        if not context.get('is_kineto_profiling', False):
            return

        self._torch_prof.stop()

    def on_span_read(self, proto, context, options):
        if not context.get('is_kineto_profiling', False):
            return

        total_self_host_time_ns = 0
        for event_avg in self._torch_prof.key_averages():
            if event_avg.key and event_avg.key.startswith('ProfilerStep'):
                continue
            op_stats = proto.op_profile.add()
            op_stats.op_type = signals_pb2.OpStats.OpType.PYTORCH_OP
            op_stats.op_name = event_avg.key
            op_stats.count = _uint(event_avg.count)
            op_stats.host_time_ns = _ns(event_avg.cpu_time_total)
            op_stats.device_time_ns = _ns(event_avg.cuda_time_total)
            op_stats.self_host_time_ns = _ns(event_avg.self_cpu_time_total)
            op_stats.self_device_time_ns = _ns(event_avg.self_cuda_time_total)
            op_stats.host_memory = _uint(event_avg.cpu_memory_usage)
            op_stats.device_memory = _uint(event_avg.cuda_memory_usage)
            op_stats.self_host_memory = _uint(event_avg.self_cpu_memory_usage)
            op_stats.self_device_memory = _uint(event_avg.self_cuda_memory_usage)
            op_stats.flops = _uint(event_avg.flops)
            total_self_host_time_ns += op_stats.self_host_time_ns

        if total_self_host_time_ns > 0:
            for op_stats in proto.op_profile:
                op_stats.self_host_time_percent = op_stats.self_host_time_ns / total_self_host_time_ns * 100

        kernel_index = {}
        for event in self._torch_prof.events():
            for kernel in event.kernels:
                key = (event.key, kernel.name, kernel.device)
                if key in kernel_index:
                    kernel_stats = kernel_index[key]
                    kernel_stats.count += 1
                    kernel_stats.duration_ns += _ns(kernel.duration)
                else:
                    kernel_stats = kernel_index[key] = signals_pb2.KernelStats()
                    kernel_stats.device_idx = kernel.device
                    kernel_stats.op_name = event.name
                    kernel_stats.kernel_name = kernel.name
                    kernel_stats.count = 1
                    kernel_stats.duration_ns = _ns(kernel.duration)

        for kernel_stats in kernel_index.values():
            proto.kernel_profile.append(kernel_stats)

        proto.labels.append('profiled')

        self._profiler_lock.release()


def _ns(val):
    return int(max(val, 0) * 1e3)


def _uint(val):
    return max(int(val), 0)
