from typing import Optional
import logging
import os
import cProfile, pstats

import graphsignal
from graphsignal.inference_span import InferenceSpan

logger = logging.getLogger('graphsignal')


import graphsignal
from graphsignal.proto import signals_pb2
from graphsignal.inference_span import InferenceSpan
from graphsignal.tracers.operation_profiler import OperationProfiler

logger = logging.getLogger('graphsignal')

class PythonProfiler(OperationProfiler):
    def __init__(self):
        self._profiler = None
        self._exclude_path = os.path.dirname(os.path.realpath(graphsignal.__file__))

    def read_info(self, signal):
        pass

    def start(self, signal, context):
        logger.debug('Activating Python profiler')

        self._profiler = cProfile.Profile()
        self._profiler.enable()

    def stop(self, signal, context):
        logger.debug('Deactivating Python profiler')

        self._profiler.disable()
        self._convert_to_operations(signal)
        self._profiler = None

    def _convert_to_operations(self, signal):
        stats = pstats.Stats(self._profiler)
        func_list = stats.fcn_list[:] if stats.fcn_list else list(stats.stats.keys())
        if not func_list:
            return

        for func in func_list:
            file_name, line_number, func_name = func
            if file_name.startswith(self._exclude_path):
                continue
            cc, nc, tt, ct, callers = stats.stats[func]

            op_stats = signal.op_stats.add()
            op_stats.device_type = signals_pb2.DeviceType.CPU
            op_stats.op_name = '{0} ({1}:{2})'.format(func_name, file_name, line_number)
            op_stats.count = int(nc)
            op_stats.total_host_time_us = _to_us(ct)
            op_stats.self_host_time_us = _to_us(tt)


def _to_us(sec):
    return int(sec * 1e6)


_profiler = PythonProfiler()


def inference_span(
        model_name: str,
        tags: Optional[dict] = None,
        ensure_trace: Optional[bool] = False) -> InferenceSpan:
    graphsignal._check_configured()

    return InferenceSpan(
        model_name=model_name,
        tags=tags,
        ensure_trace=ensure_trace,
        operation_profiler=_profiler)
