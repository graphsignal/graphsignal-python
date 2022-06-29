from typing import Optional
import logging
import os
import sys
import time
import json
import tensorflow as tf

import graphsignal
from graphsignal.proto_utils import parse_semver, compare_semver
from graphsignal.proto import profiles_pb2
from graphsignal.profiling_step import ProfilingStep
from graphsignal.profilers.operation_profiler import OperationProfiler
from graphsignal.profilers.profiler_utils import create_log_dir, remove_log_dir, convert_tensorflow_profile

logger = logging.getLogger('graphsignal')

class TensorflowProfiler(OperationProfiler):
    def __init__(self):
        self._is_initialized = False
        self._log_dir = None
        self._tensorflow_version = None
        self._global_rank = None
        self._world_size = None

    def start(self, profile):
        logger.debug('Activating TensorFlow profiler')

        # Initialization
        if not self._is_initialized:
            self._is_initialized = True

            self._tensorflow_version = profiles_pb2.SemVer()
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

        # Profiler info
        profile.profiler_info.operation_profiler_type = profiles_pb2.ProfilerInfo.ProfilerType.TENSORFLOW_PROFILER

        # Framework info
        framework = profile.frameworks.add()
        framework.type = profiles_pb2.FrameworkInfo.FrameworkType.TENSORFLOW_FRAMEWORK
        framework.version.CopyFrom(self._tensorflow_version)

        # Process info
        if self._global_rank is not None and self._global_rank >= 0:
            if graphsignal._agent.global_rank == -1:
                profile.process_usage.global_rank = self._global_rank

        # Step stats
        if self._world_size is not None and self._world_size > 0:
            profile.step_stats.world_size = self._world_size
            graphsignal.log_parameter('world_size', self._world_size)

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

    def stop(self, profile):
        logger.debug('Deactivating TensorFlow profiler')

        try:
            tf.profiler.experimental.stop()

            convert_tensorflow_profile(self._log_dir, profile)
        except Exception as e:
            raise e
        finally:
           remove_log_dir(self._log_dir)


_profiler = TensorflowProfiler()

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
