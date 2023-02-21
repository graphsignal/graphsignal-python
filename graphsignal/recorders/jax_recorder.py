import logging
import jax

import graphsignal
from graphsignal.recorders.base_recorder import BaseRecorder
from graphsignal.proto_utils import parse_semver
from graphsignal.proto import signals_pb2

logger = logging.getLogger('graphsignal')

class JAXRecorder(BaseRecorder):
    def __init__(self):
        self._framework = None

    def setup(self):
        self._framework = signals_pb2.FrameworkInfo()
        self._framework.name = 'JAX'
        parse_semver(self._framework.version, jax.__version__)

    def on_trace_start(self, signal, context, options):
        pass

    def on_trace_stop(self, signal, context, options):
        pass

    def on_trace_read(self, signal, context, options):
        pass

    def on_trace_read(self, signal, context, options):
        if self._framework:
            signal.frameworks.append(self._framework)
