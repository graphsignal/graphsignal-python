from typing import Optional
import logging
import os
import sys
import time
import gzip
import jax

import graphsignal
from graphsignal.proto_utils import parse_semver, compare_semver
from graphsignal.proto import profiles_pb2
from graphsignal.profiling_step import ProfilingStep
from graphsignal.profilers.operation_profiler import OperationProfiler
from graphsignal.profilers.profiler_utils import create_log_dir, remove_log_dir, convert_tensorflow_profile

logger = logging.getLogger('graphsignal')

class JaxProfiler(OperationProfiler):
    def __init__(self):
        self._is_initialized = False
        self._log_dir = None
        self._jax_version = None

    def start(self, profile):
        logger.debug('Activating JAX profiler')

        # Initialization
        if not self._is_initialized:
            self._is_initialized = True

            self._jax_version = profiles_pb2.SemVer()
            parse_semver(self._jax_version, jax.__version__)

        # Profiler info
        profile.profiler_info.operation_profiler_type = profiles_pb2.ProfilerInfo.ProfilerType.JAX_PROFILER

        # Framework info
        framework = profile.frameworks.add()
        framework.type = profiles_pb2.FrameworkInfo.FrameworkType.JAX_FRAMEWORK
        framework.version.CopyFrom(self._jax_version)

        try:
            self._log_dir = create_log_dir()

            jax.profiler.start_trace(self._log_dir)
        except Exception as e:
            remove_log_dir(self._log_dir)
            raise e

    def stop(self, profile):
        logger.debug('Deactivating JAX profiler')

        try:
            jax.profiler.stop_trace()

            convert_tensorflow_profile(self._log_dir, profile)
        except Exception as e:
            raise e
        finally:
            remove_log_dir(self._log_dir)


_profiler = JaxProfiler()

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
