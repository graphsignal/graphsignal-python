import logging
import sys
import time
import autogpt

import graphsignal
from graphsignal.spans import TraceOptions
from graphsignal.recorders.base_recorder import BaseRecorder
from graphsignal.recorders.instrumentation import instrument_method, uninstrument_method, read_args
from graphsignal.proto_utils import parse_semver, compare_semver
from graphsignal.proto import signals_pb2
from graphsignal.proto_utils import add_library_param, add_driver

logger = logging.getLogger('graphsignal')


class AutoGPTRecorder(BaseRecorder):
    def __init__(self):
        self._library = None
        self._is_instrumented_get_relevant = False

    def setup(self):
        if not graphsignal._tracer.auto_instrument:
            return

        self._library = signals_pb2.LibraryInfo()
        self._library.name = 'AutoGPT'

        instrument_method(autogpt.agent.agent, 'chat_with_ai', 'autogpt.chat.chat_with_ai', trace_func=self.trace_chat_with_ai)
        instrument_method(autogpt.agent.agent, 'execute_command', 'autogpt.app.execute_command', trace_func=self.trace_execute_command)

    def shutdown(self):
        uninstrument_method(autogpt.agent.agent, 'chat_with_ai')

    def trace_chat_with_ai(self, span, args, kwargs, ret, exc):
        params = read_args(args, kwargs, ['prompt', 'user_input', 'full_message_history', 'permanent_memory', 'token_limit'])

        span.set_tag('component', 'Agent')

        if not self._is_instrumented_get_relevant:
            self._is_instrumented_get_relevant = True
            if params['permanent_memory']:
                instrument_method(params['permanent_memory'], 
                    'get_relevant',
                    params['permanent_memory'].__class__.__name__ + '.get_relevant',
                    trace_func=self.trace_memory_get_relevant)
                instrument_method(params['permanent_memory'], 
                    'add',
                    params['permanent_memory'].__class__.__name__ + '.add',
                    trace_func=self.trace_memory_add)

        if 'token_limit' in params:
            span.set_param('token_limit', params['token_limit'])

    def trace_execute_command(self, span, args, kwargs, ret, exc):
        params = read_args(args, kwargs, ['command', 'arguments'])

        span.set_tag('component', 'Tool')

        if 'command' in params:
            span.set_param('command', params['command'])
        if 'arguments' in params:
            span.set_data('arguments', params['arguments'])

    def trace_memory_get_relevant(self, span, args, kwargs, ret, exc):
        params = read_args(args, kwargs, ['data', 'num_relevant'])

        span.set_tag('component', 'Memory')

        if 'data' in params:
            span.set_data('data', params['data'])
        if ret:
            span.set_data('result', ret)
        if 'num_relevant' in params:
            span.set_param('num_relevant', params['num_relevant'])

    def trace_memory_add(self, span, args, kwargs, ret, exc):
        params = read_args(args, kwargs, ['data'])

        span.set_tag('component', 'Memory')

        if 'data' in params:
            span.set_data('data', params['data'])

    def on_span_read(self, proto, context, options):
        if self._library:
            proto.libraries.append(self._library)
