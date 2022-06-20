from typing import Optional
import logging
import os
import cProfile, pstats

import graphsignal
from graphsignal.profiling_step import ProfilingStep

logger = logging.getLogger('graphsignal')


import graphsignal
from graphsignal.proto import profiles_pb2
from graphsignal.profiling_step import ProfilingStep
from graphsignal.profilers.operation_profiler import OperationProfiler

logger = logging.getLogger('graphsignal')

class GenericProfiler(OperationProfiler):
    def __init__(self):
        self._profiler = None
        self._exclude_path = os.path.dirname(os.path.realpath(graphsignal.__file__))

    def start(self, profile):
        logger.debug('Activating generic profiler')

        # Profiler info
        profile.profiler_info.operation_profiler_type = profiles_pb2.ProfilerInfo.ProfilerType.GENERIC_PROFILER

        self._profiler = cProfile.Profile()
        self._profiler.enable()

    def stop(self, profile):
        logger.debug('Deactivating generic profiler')

        self._profiler.disable()
        self._convert_to_operations(profile)
        self._profiler = None

    def _convert_to_operations(self, profile):
        stats = pstats.Stats(self._profiler)
        func_list = stats.fcn_list[:] if stats.fcn_list else list(stats.stats.keys())
        if not func_list:
            return

        for func in func_list:
            file_name, line_number, func_name = func
            if file_name.startswith(self._exclude_path):
                continue
            cc, nc, tt, ct, callers = stats.stats[func]

            op_stats = profile.op_stats.add()
            op_stats.device_type = profiles_pb2.DeviceType.CPU
            op_stats.op_name = '{0} ({1}:{2})'.format(func_name, file_name, line_number)
            op_stats.count = int(nc)
            op_stats.total_host_time_us = _to_us(ct)
            op_stats.self_host_time_us = _to_us(tt)


def _to_us(sec):
    return int(sec * 1e6)


_profiler = GenericProfiler()


def profile_step(
        phase_name: Optional[str] = None,
        effective_batch_size: Optional[int] = None,
        ensure_profile: Optional[bool] = False) -> ProfilingStep:
    graphsignal._check_configured()

    return ProfilingStep(
        phase_name=phase_name,
        effective_batch_size=effective_batch_size,
        ensure_profile=ensure_profile,
        operation_profiler=_profiler)
