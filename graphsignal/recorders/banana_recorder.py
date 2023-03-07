import logging
import sys
import time
import banana_dev as banana

import graphsignal
from graphsignal.endpoint_trace import TraceOptions
from graphsignal.recorders.base_recorder import BaseRecorder
from graphsignal.recorders.instrumentation import instrument_method, uninstrument_method, read_args
from graphsignal.proto_utils import parse_semver, compare_semver
from graphsignal.proto import signals_pb2
from graphsignal.proto_utils import add_framework_param, add_driver

logger = logging.getLogger('graphsignal')

class BananaRecorder(BaseRecorder):
    def __init__(self):
        self._framework = None
        self._is_sampling = False

    def setup(self):
        if not graphsignal._agent.auto_instrument:
            return

        self._framework = signals_pb2.FrameworkInfo()
        self._framework.name = 'Banana Python SDK'

        instrument_method(banana, 'run', 'banana.run', self.trace_run)

    def shutdown(self):
        uninstrument_method(banana, 'run', 'banana.run')

    def trace_run(self, trace, args, kwargs, ret, exc):
        params = read_args(args, kwargs, ['api_key', 'model_key', 'model_inputs'])

        if self._is_sampling:
            if 'model_key' in params:
                trace.set_param('model_key', params['model_key'])

        if 'model_inputs' in params:
            trace.set_data('model_inputs', params['model_inputs'])

        if ret and 'modelOutputs' in ret:
            trace.set_data('model_outputs', ret['modelOutputs'])

    def on_trace_start(self, signal, context, options):
        self._is_sampling = True

    def on_trace_stop(self, signal, context, options):
        self._is_sampling = False

    def on_trace_read(self, signal, context, options):
        if self._framework:
            signal.frameworks.append(self._framework)
