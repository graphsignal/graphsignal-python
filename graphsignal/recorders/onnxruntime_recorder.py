import logging
import onnxruntime

import graphsignal
from graphsignal.recorders.base_recorder import BaseRecorder
from graphsignal.proto_utils import parse_semver
from graphsignal.proto import signals_pb2

logger = logging.getLogger('graphsignal')

class ONNXRuntimeRecorder(BaseRecorder):
    def __init__(self):
        self._framework = None

    def setup(self):
        self._framework = signals_pb2.FrameworkInfo()
        self._framework.name = 'ONNX Runtime'
        parse_semver(self._framework.version, onnxruntime.__version__)

    def on_trace_start(self, proto, context, options):
        pass

    def on_trace_stop(self, proto, context, options):
        pass

    def on_trace_read(self, proto, context, options):
        if self._framework:
            proto.frameworks.append(self._framework)

    def on_metric_update(self):
        pass
