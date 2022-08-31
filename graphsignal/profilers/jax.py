from typing import Optional
import logging
import os
import sys
import time
import gzip
import jax

import graphsignal
from graphsignal.proto_utils import parse_semver, compare_semver
from graphsignal.proto import signals_pb2
from graphsignal.profilers.operation_profiler import OperationProfiler
from graphsignal.profilers.profiler_utils import create_log_dir, remove_log_dir, convert_tensorflow_profile

logger = logging.getLogger('graphsignal')

class JaxProfiler(OperationProfiler):
    def __init__(self):
        self._log_dir = None
        self._jax_version = None

    def read_info(self, signal):
        if not self._jax_version:
            self._jax_version = signals_pb2.SemVer()
            parse_semver(self._jax_version, jax.__version__)

        # Framework info
        framework = signal.frameworks.add()
        framework.type = signals_pb2.FrameworkInfo.FrameworkType.JAX_FRAMEWORK
        framework.version.CopyFrom(self._jax_version)

    def start(self, signal):
        logger.debug('Activating JAX profiler')

        # Profiler info
        signal.agent_info.operation_profiler_type = signals_pb2.AgentInfo.ProfilerType.JAX_PROFILER

        try:
            self._log_dir = create_log_dir()

            jax.profiler.start_trace(self._log_dir)
        except Exception as e:
            remove_log_dir(self._log_dir)
            raise e

    def stop(self, signal):
        logger.debug('Deactivating JAX profiler')

        try:
            jax.profiler.stop_trace()

            convert_tensorflow_profile(self._log_dir, signal)
        finally:
            remove_log_dir(self._log_dir)
