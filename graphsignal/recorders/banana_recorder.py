import logging
import sys
import time
import banana_dev as banana

import graphsignal
from graphsignal.spans import TraceOptions
from graphsignal.recorders.base_recorder import BaseRecorder
from graphsignal.recorders.instrumentation import instrument_method, uninstrument_method, read_args
from graphsignal.proto_utils import parse_semver, compare_semver
from graphsignal.proto import signals_pb2
from graphsignal.proto_utils import add_framework_param, add_driver

logger = logging.getLogger('graphsignal')

class BananaRecorder(BaseRecorder):
    def __init__(self):
        self._framework = None

    def setup(self):
        if not graphsignal._tracer.auto_instrument:
            return

        self._framework = signals_pb2.FrameworkInfo()
        self._framework.name = 'Banana Python SDK'

        instrument_method(banana, 'run', 'banana.run', self.trace_run)

    def shutdown(self):
        uninstrument_method(banana, 'run', 'banana.run')

    def trace_run(self, span, args, kwargs, ret, exc):
        params = read_args(args, kwargs, ['api_key', 'model_key', 'model_inputs'])

        span.set_tag('component', 'Model')
        span.set_tag('endpoint', 'https://api.banana.dev')

        if 'model_key' in params:
            span.set_tag('model_key', params['model_key'])
            span.set_param('model_key', params['model_key'])

        if 'model_inputs' in params:
            span.set_data('model_inputs', params['model_inputs'])

        if ret:
            span.set_data('model_outputs', ret)

    def on_span_read(self, proto, context, options):
        if self._framework:
            proto.frameworks.append(self._framework)
