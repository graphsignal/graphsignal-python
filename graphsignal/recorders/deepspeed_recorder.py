import logging
import sys
import time
import torch
import deepspeed
import deepspeed.comm as dist
import threading

import graphsignal
from graphsignal.traces import TraceOptions
from graphsignal.recorders.base_recorder import BaseRecorder
from graphsignal.recorders.instrumentation import patch_method
from graphsignal.proto_utils import parse_semver, compare_semver
from graphsignal.proto import signals_pb2
from graphsignal.proto_utils import add_framework_param, add_driver

logger = logging.getLogger('graphsignal')

class DeepSpeedRecorder(BaseRecorder):
    COLLECTIVE_OPS = [
        'broadcast',
        'all_gather',
        'reduce_scatter_base',
        'all_gather_base',
        'all_to_all_single',
        'send',
        'recv',
        'isend',
        'irecv',
        'gather',
        'scatter',
        'barrier',
        'monitored_barrier',
        'reduce',
        'reduce_scatter',
        'all_reduce'
    ]

    def __init__(self):
        self._profiler_lock = threading.Lock()
        self._is_initialized = False
        self._framework = None
        self._rank = None
        self._local_rank = None
        self._world_size = None
        self._is_sampling = False
        self._op_profile = None

    def setup(self):
        self._framework = signals_pb2.FrameworkInfo()
        self._framework.name = 'DeepSpeed'
        parse_semver(self._framework.version, deepspeed.__version__)

        if dist.is_initialized():
            add_framework_param(self._framework, 'deepspeed.comm.get_world_size', dist.get_world_size())
            add_framework_param(self._framework, 'deepspeed.comm.get_rank', dist.get_rank())
            add_framework_param(self._framework, 'deepspeed.comm.get_local_rank', dist.get_local_rank())

            self._rank = dist.get_rank()
            self._local_rank = dist.get_local_rank()
            self._world_size = dist.get_world_size()

            if compare_semver(self._framework.version, (0, 7, 0)) < 1:
                logger.debug('DeepSpeed tracing is only supported for >= 0.7.0.')
                return

            # auto-instrument collective operations to be traced as standalone
            for op_name in DeepSpeedRecorder.COLLECTIVE_OPS:
                self._instrument_op(op_name)

            self._is_initialized = True

    def _instrument_op(self, op_name):
        def before_op(args, kwargs):
            if self._is_sampling:
                return time.perf_counter_ns()
            return None

        def after_op(args, kwargs, ret, exc, start_counter):
            if self._is_sampling and start_counter is not None:
                stop_counter = time.perf_counter_ns()
                latency_us = int((stop_counter - start_counter) / 1e3)
                data_size = self._get_data_size(op_name, args, kwargs)

                if not self._op_profile:
                    self._op_profile = {}
                if op_name not in self._op_profile:
                    self._op_profile[op_name] = {
                        'count': 1,
                        'total_time_us': latency_us,
                        'total_data_size': data_size
                    }
                else:
                    op_stats = self._op_profile[op_name]
                    op_stats['count'] += 1
                    op_stats['total_time_us'] += latency_us
                    op_stats['total_data_size'] += data_size

        if not patch_method(dist, op_name, before_func=before_op, after_func=after_op):
            logger.debug('Cannot instrument DeepSpeed communications logger.')

    def on_trace_start(self, proto, context, options):
        if not self._is_initialized:
            return
        if not self._profiler_lock.acquire(blocking=False):
            return

        self._is_sampling = True

    def on_trace_stop(self, proto, context, options):
        if not self._is_initialized:
            return

        self._is_sampling = False

    def on_trace_read(self, proto, context, options):
        if not self._is_initialized:
            return

        if self._framework:
            proto.frameworks.append(self._framework)
        if self._rank is not None:
            proto.process_usage.rank = self._rank
            proto.process_usage.has_rank = True            
        if self._local_rank is not None:
            proto.process_usage.local_rank = self._local_rank
            proto.process_usage.has_local_rank = True

        if self._op_profile:
            for name, stats in self._op_profile.items():
                op_stats = proto.op_profile.add()
                op_stats.op_type = signals_pb2.OpStats.OpType.COLLECTIVE_OP
                op_stats.op_name = name
                op_stats.count = stats['count']
                op_stats.host_time_ns = int(stats['total_time_us'] * 1e3)
                op_stats.data_size = stats['total_data_size']
                if op_stats.host_time_ns > 0:
                    op_stats.data_per_sec = op_stats.data_size / (op_stats.host_time_ns / 1e9)

            self._op_profile = None

        self._profiler_lock.release()

    def _get_data_size(self, op_name, args, kwargs):
        tensor_arg = None

        if len(args) > 0:
            tensor_arg = args[0]
        elif len(kwargs) > 0:
            if 'tensor' in kwargs:
                tensor_arg = kwargs['tensor']
            elif 'output_tensor' in kwargs:
                tensor_arg = kwargs['output_tensor']
            elif 'output' in kwargs:
                tensor_arg = kwargs['output']
            elif 'tensor_list' in kwargs:
                tensor_arg = kwargs['tensor_list']

        msg_size = 0
        if isinstance(tensor_arg, torch.Tensor):
            msg_size = tensor_arg.element_size() * tensor_arg.nelement()
        elif isinstance(tensor_arg, list) and len(tensor_arg) > 0 and isinstance(tensor_arg[0], torch.Tensor):
            msg_size = sum(el.element_size() * el.nelement() for el in tensor_arg)
        else:
            return 0

        if op_name in ("all_gather", "all_gather_base", "reduce_scatter", "reduce_scatter_base"):
            return msg_size * self._world_size
        elif op_name == "all_reduce":
            return msg_size * 2
        else:
            return msg_size

