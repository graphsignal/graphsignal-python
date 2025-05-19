import logging
import os
import json
import time
import torch

import graphsignal
from graphsignal.recorders.base_recorder import BaseRecorder
from graphsignal.recorders.profiler_utils import create_log_dir, remove_log_dir

logger = logging.getLogger('graphsignal')

class PyTorchRecorder(BaseRecorder):
    def __init__(self):
        self._torch_prof = None
        self._log_dir = None

    def setup(self):
        pass

    def _can_include_profiles(self, span, profiles):
        return (graphsignal._tracer.can_include_profiles(profiles) and 
                span.can_include_profiles(profiles))

    def on_span_start(self, span, context):
        if (span.sampled() and 
            self._can_include_profiles(span, ['pytorch-cpu-profile', 'pytorch-kernel-profile', 'pytorch-timeline']) and 
            graphsignal._tracer.set_profiling_mode()):
            context['profiled'] = True

            if self._torch_prof:
                # In case of previous profiling not stopped
                self._torch_prof.stop()
                self._torch_prof = None

            def _schedule_func(step):
                return torch.profiler.ProfilerAction.RECORD
            self._torch_prof = torch.profiler.profile(
                schedule=_schedule_func,
                record_shapes=False,
                profile_memory=True,
                with_stack=False,
                with_flops=True)
            self._torch_prof.start()

    def on_span_stop(self, span, context):
        if context.get('profiled', False):
            graphsignal._tracer.unset_profiling_mode()
            if self._torch_prof:
                self._torch_prof.stop()

    def on_span_read(self, span, context):
        span.set_param('pytorch_version', torch.__version__)

        if context.get('profiled', False):
            if self._torch_prof:
                try:
                    self._convert_to_profile(span)
                finally:
                    self._torch_prof = None
            span.set_param('profiler', f'pytorch-{torch.__version__}')

    def _convert_to_profile(self, span):
        if self._can_include_profiles(span, ['pytorch-cpu-profile']):
            cpu_profile = []
            for event_avg in self._torch_prof.key_averages():
                if event_avg.key and event_avg.key.startswith('ProfilerStep'):
                    continue
                cpu_profile.append(dict(
                    op_name = event_avg.key,
                    device_type = event_avg.device_type.name if event_avg.device_type else None,
                    count = _uint(event_avg.count),
                    cpu_time_ns = _ns(event_avg.cpu_time_total),
                    self_cpu_time_ns = _ns(event_avg.self_cpu_time_total),
                    device_time_ns = _ns(event_avg.device_time_total),
                    self_device_time_ns = _ns(event_avg.self_device_time_total),
                    cpu_memory = _uint(event_avg.cpu_memory_usage),
                    self_cpu_memory = _uint(event_avg.self_cpu_memory_usage),
                    device_memory = _uint(event_avg.device_memory_usage),
                    self_device_memory = _uint(event_avg.self_device_memory_usage),
                    flops = _uint(event_avg.flops)
                ))

            if len(cpu_profile) > 0:
                span.set_profile(
                    name='pytorch-cpu-profile', 
                    format='event-averages', 
                    content=json.dumps(cpu_profile))

        if self._can_include_profiles(span, ['pytorch-kernel-profile']):
            kernel_index = {}
            for event in self._torch_prof.events():
                for kernel in event.kernels:
                    key = (event.key, kernel.name, kernel.device)
                    if key in kernel_index:
                        kernel_avg = kernel_index[key]
                        kernel_avg['count'] += 1
                        kernel_avg['duration_ns'] += _ns(kernel.duration)
                    else:
                        kernel_avg = kernel_index[key] = dict(
                            device_idx = kernel.device,
                            op_name = event.name,
                            kernel_name = kernel.name,
                            count = 1,
                            duration_ns = _ns(kernel.duration)
                        )
            
            device_profile = kernel_index.values()
            if len(device_profile) > 0:
                span.set_profile(
                    name='pytorch-kernel-profile', 
                    format='event-averages', 
                    content=json.dumps(device_profile))

        if self._can_include_profiles(span, ['pytorch-timeline']):
            if len(cpu_profile) > 0 or len(device_profile) > 0:
                chrome_trace = self._export_chrome_trace()
                if chrome_trace:
                    span.set_profile('pytorch-timeline', 'chrome-trace', chrome_trace)

    def _export_chrome_trace(self):
        try:
            read_start_time = time.time()

            self._log_dir = create_log_dir()

            trace_path = os.path.join(self._log_dir, 'trace.json')
            self._torch_prof.export_chrome_trace(trace_path)

            trace_file_size = os.path.getsize(trace_path)
            logger.debug('Chrome trace size: %s', trace_file_size)
            if trace_file_size > 50 * 1e6:
                logger.debug('Chrome trace file too big: %s', trace_file_size)
                return None

            with open(trace_path, "r") as f:
                return str(f.read())
        finally:
            remove_log_dir(self._log_dir)
            logger.debug('Chrome trace export time: %s', time.time() - read_start_time)

        return None

def _ns(val):
    return int(max(val, 0) * 1e3)


def _uint(val):
    return max(int(val), 0)
