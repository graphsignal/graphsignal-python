import logging
import os
import json
import tensorflow as tf

import graphsignal
from graphsignal.recorders.base_recorder import BaseRecorder
from graphsignal.proto_utils import parse_semver
from graphsignal.proto import signals_pb2
from graphsignal.proto_utils import add_framework_param

logger = logging.getLogger('graphsignal')

class TensorFlowRecorder(BaseRecorder):
    def __init__(self):
        self._framework = None
        self._comm_info = None

    def setup(self):
        self._framework = signals_pb2.FrameworkInfo()
        self._framework.type = signals_pb2.FrameworkInfo.FrameworkType.TENSORFLOW_FRAMEWORK
        parse_semver(self._framework.version, tf.__version__)

        if 'TF_CONFIG' in os.environ:
            try:
                tf_config = json.loads(os.environ['TF_CONFIG'])

                cluster_size = 0
                if 'chief' in tf_config['cluster']:
                    cluster_size += len(tf_config['cluster']['chief'])
                if 'worker' in tf_config['cluster']:
                    cluster_size += len(tf_config['cluster']['worker'])
                if cluster_size > 0:
                    add_framework_param(self._framework, 'cluster_size', cluster_size)

                add_framework_param(self._framework, 'task_index', tf_config['task']['index'])

            except:
                logger.warning('Error parsing TF_CONFIG', exc_info=True)

        add_framework_param(self._framework, 'tf.test.is_built_with_gpu_support', tf.test.is_built_with_gpu_support())
        add_framework_param(self._framework, 'tf.test.is_built_with_cuda', tf.test.is_built_with_cuda())

    def on_trace_start(self, signal, context):
        pass

    def on_trace_stop(self, signal, context):
        pass

    def on_trace_read(self, signal, context):
        if self._framework:
            signal.frameworks.append(self._framework)
