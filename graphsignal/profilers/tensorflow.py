from typing import Optional
import logging
import os
import sys
import time
import json
import tensorflow as tf

import graphsignal
from graphsignal.proto_utils import parse_semver, compare_semver
from graphsignal.proto import signals_pb2
from graphsignal.profilers.operation_profiler import OperationProfiler
from graphsignal.profilers.profiler_utils import create_log_dir, remove_log_dir, convert_tensorflow_profile

logger = logging.getLogger('graphsignal')

class TensorFlowProfiler(OperationProfiler):
    def __init__(self):
        self._log_dir = None
        self._tensorflow_version = None

    def read_info(self, signal):
        if not self._tensorflow_version:
            self._tensorflow_version = signals_pb2.SemVer()
            parse_semver(self._tensorflow_version, tf.__version__)
            if compare_semver(self._tensorflow_version, (2, 2, 0)) == -1:
                raise Exception(
                    'TensorFlow profiling is not supported for versions <=2.2')

        # Framework info
        framework = signal.frameworks.add()
        framework.type = signals_pb2.FrameworkInfo.FrameworkType.TENSORFLOW_FRAMEWORK
        framework.version.CopyFrom(self._tensorflow_version)

    def start(self, signal):
        logger.debug('Activating TensorFlow profiler')

        # Profiler info
        signal.agent_info.operation_profiler_type = signals_pb2.AgentInfo.ProfilerType.TENSORFLOW_PROFILER

        try:
            self._log_dir = create_log_dir()

            options = tf.profiler.experimental.ProfilerOptions(
                host_tracer_level=2,
                python_tracer_level=0,
                device_tracer_level=1)
            tf.profiler.experimental.start(self._log_dir, options=options)
        except Exception as e:
            remove_log_dir(self._log_dir)
            raise e

    def stop(self, signal):
        logger.debug('Deactivating TensorFlow profiler')

        try:
            tf.profiler.experimental.stop()

            convert_tensorflow_profile(self._log_dir, signal)
        finally:
           remove_log_dir(self._log_dir)
