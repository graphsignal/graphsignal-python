import logging
import os
import time
import torch

import graphsignal
from graphsignal.recorders.base_recorder import BaseRecorder
from graphsignal.utils import create_log_dir, remove_log_dir

logger = logging.getLogger('graphsignal')


class PyTorchRecorder(BaseRecorder):
    def __init__(self):
        self._torch_prof = None
        self._log_dir = None

    def setup(self):
        ticker = graphsignal._ticker
        ticker.set_tag('framework.name', 'pytorch')
        ticker.set_tag('framework.version', torch.__version__)

        for category, function_path in PROFILED_PATHS:
            ticker.profile_function_path(function_path, category=category)

    def on_tick(self):
        if not torch.cuda.is_available():
            return

        now_ns = time.time_ns()
        device_count = torch.cuda.device_count()
        
        for device_idx in range(device_count):
            try:
                # Get device properties
                device_props = torch.cuda.get_device_properties(device_idx)
                device_name = device_props.name
                
                # Get memory stats for this device
                memory_stats = torch.cuda.memory_stats(device_idx)
                
                # Set up metric tags
                ticker = graphsignal._ticker

                metric_tags = {}
                metric_tags['device.type'] = 'gpu'
                metric_tags['device.index'] = device_idx
                metric_tags['device.name'] = device_name
                metric_tags['framework.name'] = 'pytorch'
                metric_tags['framework.version'] = torch.__version__
                
                # Record memory metrics
                if 'allocated_bytes.all.current' in memory_stats:
                    ticker.set_gauge(
                        name='pytorch.memory.allocated', tags=metric_tags,
                        value=memory_stats['allocated_bytes.all.current'], measurement_ts=now_ns)
                
                if 'reserved_bytes.all.current' in memory_stats:
                    ticker.set_gauge(
                        name='pytorch.memory.reserved', tags=metric_tags,
                        value=memory_stats['reserved_bytes.all.current'], measurement_ts=now_ns)
                
                if 'allocated_bytes.all.peak' in memory_stats:
                    ticker.set_gauge(
                        name='pytorch.memory.allocated.peak', tags=metric_tags,
                        value=memory_stats['allocated_bytes.all.peak'], measurement_ts=now_ns)
                
                if 'reserved_bytes.all.peak' in memory_stats:
                    ticker.set_gauge(
                        name='pytorch.memory.reserved.peak', tags=metric_tags,
                        value=memory_stats['reserved_bytes.all.peak'], measurement_ts=now_ns)
                
                # Record allocation/deallocation counts
                if 'allocated_bytes.all.count' in memory_stats:
                    ticker.set_gauge(
                        name='pytorch.memory.allocations', tags=metric_tags,
                        value=memory_stats['allocated_bytes.all.count'], measurement_ts=now_ns)
                
                if 'allocated_bytes.all.freed' in memory_stats:
                    ticker.set_gauge(
                        name='pytorch.memory.deallocations', tags=metric_tags,
                        value=memory_stats['allocated_bytes.all.freed'], measurement_ts=now_ns)
                
                # Record additional memory management metrics
                if 'num_alloc_retries' in memory_stats:
                    ticker.set_gauge(
                        name='pytorch.memory.alloc_retries', tags=metric_tags,
                        value=memory_stats['num_alloc_retries'], measurement_ts=now_ns)
                
                if 'num_ooms' in memory_stats:
                    ticker.set_gauge(
                        name='pytorch.memory.ooms', tags=metric_tags,
                        value=memory_stats['num_ooms'], measurement_ts=now_ns)
                
                if 'num_sync_all_streams' in memory_stats:
                    ticker.set_gauge(
                        name='pytorch.memory.sync_all_streams', tags=metric_tags,
                        value=memory_stats['num_sync_all_streams'], measurement_ts=now_ns)
                
                if 'num_device_alloc' in memory_stats:
                    ticker.set_gauge(
                        name='pytorch.memory.device_alloc', tags=metric_tags,
                        value=memory_stats['num_device_alloc'], measurement_ts=now_ns)
                
                if 'num_device_free' in memory_stats:
                    ticker.set_gauge(
                        name='pytorch.memory.device_free', tags=metric_tags,
                        value=memory_stats['num_device_free'], measurement_ts=now_ns)
                
                # Record fragmentation metrics
                if 'allocated_bytes.all.peak' in memory_stats and 'reserved_bytes.all.peak' in memory_stats:
                    if memory_stats['reserved_bytes.all.peak'] > 0:
                        fragmentation = (memory_stats['reserved_bytes.all.peak'] - memory_stats['allocated_bytes.all.peak']) / memory_stats['reserved_bytes.all.peak']
                        ticker.set_gauge(
                            name='pytorch.memory.fragmentation', tags=metric_tags,
                            value=fragmentation * 100, measurement_ts=now_ns, unit='percent')
                
                # Record cache metrics
                if 'allocated_bytes.all.cached' in memory_stats:
                    ticker.set_gauge(
                        name='pytorch.memory.cached', tags=metric_tags,
                        value=memory_stats['allocated_bytes.all.cached'], measurement_ts=now_ns)
                
                # Record device memory info
                device_memory = torch.cuda.get_device_properties(device_idx).total_memory
                if device_memory > 0:
                    ticker.set_gauge(
                        name='pytorch.memory.total', tags=metric_tags,
                        value=device_memory, measurement_ts=now_ns)
                
                # Calculate memory utilization percentage
                if device_memory > 0 and 'reserved_bytes.all.current' in memory_stats:
                    utilization = (memory_stats['reserved_bytes.all.current'] / device_memory) * 100
                    ticker.set_gauge(
                        name='pytorch.memory.utilization', tags=metric_tags,
                        value=utilization, measurement_ts=now_ns, unit='percent')
                
            except Exception as e:
                logger.warning(f'Failed to record PyTorch memory metrics for device {device_idx}: {e}')

    def _export_chrome_trace(self):
        try:
            read_start_time = time.time()

            if len(self._torch_prof.key_averages()) == 0:
                logger.debug('PyTorch profiler has no results to export')
                return None

            self._log_dir = create_log_dir()

            trace_path = os.path.join(self._log_dir, 'trace.json')
            self._torch_prof.export_chrome_trace(trace_path)

            trace_file_size = os.path.getsize(trace_path)
            logger.debug('Chrome trace size: %s', trace_file_size)
            if trace_file_size > 500 * 1e6:
                logger.debug('Chrome trace file too big: %s', trace_file_size)
                return None

            with open(trace_path, "r") as f:
                return str(f.read())
        finally:
            if self._log_dir:
                remove_log_dir(self._log_dir)
            logger.debug('Chrome trace export time: %s', time.time() - read_start_time)

        return None

PROFILED_PATHS = [
    # NN module forwards (typically dominate compute; ms-ish depending on shapes)
    ('pytorch.nn', "torch.nn.Linear.forward"),
    ('pytorch.nn', "torch.nn.Conv1d.forward"),
    ('pytorch.nn', "torch.nn.Conv2d.forward"),
    ('pytorch.nn', "torch.nn.Conv3d.forward"),
    ('pytorch.nn', "torch.nn.ConvTranspose1d.forward"),
    ('pytorch.nn', "torch.nn.ConvTranspose2d.forward"),
    ('pytorch.nn', "torch.nn.ConvTranspose3d.forward"),
    ('pytorch.nn', "torch.nn.Embedding.forward"),
    ('pytorch.nn', "torch.nn.LayerNorm.forward"),
    ('pytorch.nn', "torch.nn.GroupNorm.forward"),
    ('pytorch.nn', "torch.nn.RMSNorm.forward"),
    ('pytorch.nn', "torch.nn.MultiheadAttention.forward"),

    # distributed collectives (Python API surface)
    ('pytorch.comm', "torch.distributed.all_reduce"),
    ('pytorch.comm', "torch.distributed.all_gather_into_tensor"),
    ('pytorch.comm', "torch.distributed.reduce_scatter_tensor"),
    ('pytorch.comm', "torch.distributed.all_to_all_single"),
    ('pytorch.comm', "torch.distributed.barrier"),
    ('pytorch.comm', "torch.distributed.send"),
    ('pytorch.comm', "torch.distributed.recv"),
    ('pytorch.comm', "torch.distributed.batch_isend_irecv"),

    # “where async comm time is paid” (functional collectives)
    ('pytorch.comm_wait', "torch.distributed._functional_collectives.wait_tensor"),
    ('pytorch.comm_wait', "torch.distributed._functional_collectives.AsyncCollectiveTensor.wait"),

    # Python-level CUDA sync points
    ('pytorch.cuda_sync', "torch.cuda.synchronize"),
    ('pytorch.cuda_sync', "torch.cuda.streams.Stream.synchronize"),
    ('pytorch.cuda_sync', "torch.cuda.streams.Event.synchronize"),
]