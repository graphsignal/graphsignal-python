import logging
import sys
import time

import graphsignal
from graphsignal.spans import TraceOptions
from graphsignal.recorders.base_recorder import BaseRecorder
from graphsignal.recorders.instrumentation import instrument_method, uninstrument_method, read_args
from graphsignal.proto_utils import parse_semver, compare_semver
from graphsignal.proto import signals_pb2
from graphsignal.proto_utils import add_library_param, add_driver

logger = logging.getLogger('graphsignal')

class BananaRecorder(BaseRecorder):
    def __init__(self):
        self._library = None

    def setup(self):
        if not graphsignal._tracer.auto_instrument:
            return

        self._library = signals_pb2.LibraryInfo()
        self._library.name = 'Banana Python SDK'

        from banana_dev import Client
        instrument_method(Client, 'call', 'banana.call', trace_func=self.trace_call)

    def shutdown(self):
        from banana_dev import Client
        uninstrument_method(Client, 'call')

    def trace_call(self, span, args, kwargs, ret, exc):
        client = args[0]
        params = read_args(args, kwargs, ['self', 'route', 'json'])

        url = client.url if client.url else ''
        endpoint = url + params.get('route', '')

        span.set_tag('component', 'Model')
        if endpoint:
            span.set_tag('endpoint', endpoint)

        if client.model_key:
            span.set_tag('model_key', client.model_key)
            span.set_param('model_key', client.model_key)

        if 'json' in params:
            span.set_data('json', params['json'])
        if 'headers' in params:
            span.set_data('headers', params['headers'])

        if isinstance(ret, tuple):
            span.set_data('output', ret[0])
            if len(ret) > 1:
                span.set_data('meta', ret[1])

    def on_span_read(self, proto, context, options):
        if self._library:
            proto.libraries.append(self._library)
