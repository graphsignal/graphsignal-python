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
        tracer = graphsignal._tracer
        tracer.set_tag('framework.name', 'pytorch')
        tracer.set_tag('framework.version', torch.__version__)

    def _can_include_profiles(self, span, profiles):
        return (graphsignal._tracer.can_include_profiles(profiles) and 
                span.can_include_profiles(profiles))

    def on_span_start(self, span, context):
        if (self._can_include_profiles(span, ['profile.pytorch']) and 
            graphsignal._tracer.set_profiling_mode('profile.pytorch')):
            context['profiled'] = True
            span.set_sampled(True)

            if self._torch_prof:
                # In case of previous profiling not stopped
                self._torch_prof.stop()
                self._torch_prof = None

            def _schedule_func(step):
                return torch.profiler.ProfilerAction.RECORD
            self._torch_prof = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU, 
                    torch.profiler.ProfilerActivity.CUDA],
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
        if context.get('profiled', False):
            if self._torch_prof:
                try:
                    self._convert_to_profile(span)
                finally:
                    self._torch_prof = None

    def on_metric_update(self):
        """Record PyTorch GPU memory metrics using torch.cuda.memory_stats"""
        if not torch.cuda.is_available():
            return

        now = int(time.time())
        device_count = torch.cuda.device_count()
        
        for device_idx in range(device_count):
            try:
                # Get device properties
                device_props = torch.cuda.get_device_properties(device_idx)
                device_name = device_props.name
                
                # Get memory stats for this device
                memory_stats = torch.cuda.memory_stats(device_idx)
                
                # Set up metric tags
                store = graphsignal._tracer.metric_store()
                metric_tags = graphsignal._tracer.tags.copy()
                metric_tags['device.type'] = 'gpu'
                metric_tags['device.index'] = device_idx
                metric_tags['device.name'] = device_name
                metric_tags['framework.name'] = 'pytorch'
                metric_tags['framework.version'] = torch.__version__
                
                # Record memory metrics
                if 'allocated_bytes.all.current' in memory_stats:
                    store.set_gauge(
                        name='pytorch.memory.allocated', tags=metric_tags,
                        value=memory_stats['allocated_bytes.all.current'], update_ts=now, is_size=True)
                
                if 'reserved_bytes.all.current' in memory_stats:
                    store.set_gauge(
                        name='pytorch.memory.reserved', tags=metric_tags,
                        value=memory_stats['reserved_bytes.all.current'], update_ts=now, is_size=True)
                
                if 'allocated_bytes.all.peak' in memory_stats:
                    store.set_gauge(
                        name='pytorch.memory.allocated.peak', tags=metric_tags,
                        value=memory_stats['allocated_bytes.all.peak'], update_ts=now, is_size=True)
                
                if 'reserved_bytes.all.peak' in memory_stats:
                    store.set_gauge(
                        name='pytorch.memory.reserved.peak', tags=metric_tags,
                        value=memory_stats['reserved_bytes.all.peak'], update_ts=now, is_size=True)
                
                # Record allocation/deallocation counts
                if 'allocated_bytes.all.count' in memory_stats:
                    store.set_gauge(
                        name='pytorch.memory.allocations', tags=metric_tags,
                        value=memory_stats['allocated_bytes.all.count'], update_ts=now)
                
                if 'allocated_bytes.all.freed' in memory_stats:
                    store.set_gauge(
                        name='pytorch.memory.deallocations', tags=metric_tags,
                        value=memory_stats['allocated_bytes.all.freed'], update_ts=now)
                
                # Record additional memory management metrics
                if 'num_alloc_retries' in memory_stats:
                    store.set_gauge(
                        name='pytorch.memory.alloc_retries', tags=metric_tags,
                        value=memory_stats['num_alloc_retries'], update_ts=now)
                
                if 'num_ooms' in memory_stats:
                    store.set_gauge(
                        name='pytorch.memory.ooms', tags=metric_tags,
                        value=memory_stats['num_ooms'], update_ts=now)
                
                if 'num_sync_all_streams' in memory_stats:
                    store.set_gauge(
                        name='pytorch.memory.sync_all_streams', tags=metric_tags,
                        value=memory_stats['num_sync_all_streams'], update_ts=now)
                
                if 'num_device_alloc' in memory_stats:
                    store.set_gauge(
                        name='pytorch.memory.device_alloc', tags=metric_tags,
                        value=memory_stats['num_device_alloc'], update_ts=now)
                
                if 'num_device_free' in memory_stats:
                    store.set_gauge(
                        name='pytorch.memory.device_free', tags=metric_tags,
                        value=memory_stats['num_device_free'], update_ts=now)
                
                # Record fragmentation metrics
                if 'allocated_bytes.all.peak' in memory_stats and 'reserved_bytes.all.peak' in memory_stats:
                    if memory_stats['reserved_bytes.all.peak'] > 0:
                        fragmentation = (memory_stats['reserved_bytes.all.peak'] - memory_stats['allocated_bytes.all.peak']) / memory_stats['reserved_bytes.all.peak']
                        store.set_gauge(
                            name='pytorch.memory.fragmentation', tags=metric_tags,
                            value=fragmentation * 100, update_ts=now, unit='percent')
                
                # Record cache metrics
                if 'allocated_bytes.all.cached' in memory_stats:
                    store.set_gauge(
                        name='pytorch.memory.cached', tags=metric_tags,
                        value=memory_stats['allocated_bytes.all.cached'], update_ts=now, is_size=True)
                
                # Record device memory info
                device_memory = torch.cuda.get_device_properties(device_idx).total_memory
                if device_memory > 0:
                    store.set_gauge(
                        name='pytorch.memory.total', tags=metric_tags,
                        value=device_memory, update_ts=now, is_size=True)
                
                # Calculate memory utilization percentage
                if device_memory > 0 and 'reserved_bytes.all.current' in memory_stats:
                    utilization = (memory_stats['reserved_bytes.all.current'] / device_memory) * 100
                    store.set_gauge(
                        name='pytorch.memory.utilization', tags=metric_tags,
                        value=utilization, update_ts=now, unit='percent')
                
            except Exception as e:
                logger.warning(f'Failed to record PyTorch memory metrics for device {device_idx}: {e}')

    def _convert_to_profile(self, span):
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
                name='profile.pytorch.cpu', 
                format='event-averages', 
                content=json.dumps(cpu_profile))

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

        device_profile = list(kernel_index.values())
        span.set_profile(
            name='profile.pytorch.kernel', 
            format='event-averages', 
            content=json.dumps(device_profile))

        chrome_trace = self._export_chrome_trace()
        if chrome_trace:
            span.set_profile('profile.pytorch.trace', 'chrome-trace', chrome_trace)

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
