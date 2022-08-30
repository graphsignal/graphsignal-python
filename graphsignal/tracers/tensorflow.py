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
from graphsignal.inference_span import InferenceSpan
from graphsignal.tracers.operation_profiler import OperationProfiler
from graphsignal.tracers.profiler_utils import create_log_dir, remove_log_dir, convert_tensorflow_profile

logger = logging.getLogger('graphsignal')

class TensorflowProfiler(OperationProfiler):
    def __init__(self):
        self._log_dir = None
        self._tensorflow_version = None
        self._global_rank = None
        self._world_size = None

    def read_info(self, signal):
        if not self._tensorflow_version:
            self._tensorflow_version = signals_pb2.SemVer()
            parse_semver(self._tensorflow_version, tf.__version__)
            if compare_semver(self._tensorflow_version, (2, 2, 0)) == -1:
                raise Exception(
                    'TensorFlow profiling is not supported for versions <=2.2')

            if 'TF_CONFIG' in os.environ:
                try:
                    tf_config = json.loads(os.environ['TF_CONFIG'])
                    self._world_size = 0
                    if 'chief' in tf_config['cluster']:
                        self._world_size += len(tf_config['cluster']['chief'])
                    if 'worker' in tf_config['cluster']:
                        self._world_size += len(tf_config['cluster']['worker'])
                    self._global_rank = tf_config['task']['index']
                except:
                    logger.warning('Error parsing TF_CONFIG', exc_info=True)

        # Framework info
        framework = signal.frameworks.add()
        framework.type = signals_pb2.FrameworkInfo.FrameworkType.TENSORFLOW_FRAMEWORK
        framework.version.CopyFrom(self._tensorflow_version)

        # Process info
        if self._global_rank is not None and self._global_rank >= 0:
            signal.process_usage.global_rank = self._global_rank

        # Cluster stats
        if self._world_size is not None and self._world_size > 0:
            signal.cluster_info.world_size = self._world_size

    def start(self, signal, context):
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

    def stop(self, signal, context):
        logger.debug('Deactivating TensorFlow profiler')

        try:
            tf.profiler.experimental.stop()

            convert_tensorflow_profile(self._log_dir, signal)
        finally:
           remove_log_dir(self._log_dir)


_profiler = TensorflowProfiler()

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
