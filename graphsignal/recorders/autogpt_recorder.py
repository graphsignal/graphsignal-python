import logging
import sys
import time
import autogpt

import graphsignal
from graphsignal.traces import TraceOptions
from graphsignal.recorders.base_recorder import BaseRecorder
from graphsignal.recorders.instrumentation import instrument_method, uninstrument_method, read_args
from graphsignal.proto_utils import parse_semver, compare_semver
from graphsignal.proto import signals_pb2
from graphsignal.proto_utils import add_framework_param, add_driver

logger = logging.getLogger('graphsignal')


class AutoGPTRecorder(BaseRecorder):
    def __init__(self):
        self._framework = None
        self._is_instrumented_get_relevant = False

    def setup(self):
        if not graphsignal._agent.auto_instrument:
            return

        self._framework = signals_pb2.FrameworkInfo()
        self._framework.name = 'AutoGPT'

        instrument_method(autogpt.agent.agent, 'chat_with_ai', 'autogpt.chat.chat_with_ai', self.trace_chat_with_ai)
        instrument_method(autogpt.agent.agent, 'execute_command', 'autogpt.app.execute_command', self.trace_execute_command)

    def shutdown(self):
        uninstrument_method(autogpt.agent.agent, 'chat_with_ai', 'autogpt.chat.chat_with_ai')

    def trace_chat_with_ai(self, trace, args, kwargs, ret, exc):
        params = read_args(args, kwargs, ['prompt', 'user_input', 'full_message_history', 'permanent_memory', 'token_limit'])

        trace.set_tag('component', 'Agent')

        if not self._is_instrumented_get_relevant:
            self._is_instrumented_get_relevant = True
            if params['permanent_memory']:
                instrument_method(params['permanent_memory'], 
                    'get_relevant',
                    params['permanent_memory'].__class__.__name__ + '.get_relevant',
                    self.trace_memory_get_relevant)
                instrument_method(params['permanent_memory'], 
                    'add',
                    params['permanent_memory'].__class__.__name__ + '.add',
                    self.trace_memory_add)

        if 'token_limit' in params:
            trace.set_param('token_limit', params['token_limit'])

    def trace_execute_command(self, trace, args, kwargs, ret, exc):
        params = read_args(args, kwargs, ['command', 'arguments'])

        trace.set_tag('component', 'Tool')

        if 'command' in params:
            trace.set_param('command', params['command'])
        if 'arguments' in params:
            trace.set_data('arguments', params['arguments'])

    def trace_memory_get_relevant(self, trace, args, kwargs, ret, exc):
        params = read_args(args, kwargs, ['data', 'num_relevant'])

        trace.set_tag('component', 'Memory')

        if 'data' in params:
            trace.set_data('data', params['data'])
        if ret:
            trace.set_data('result', ret)
        if 'num_relevant' in params:
            trace.set_param('num_relevant', params['num_relevant'])

    def trace_memory_add(self, trace, args, kwargs, ret, exc):
        params = read_args(args, kwargs, ['data'])

        trace.set_tag('component', 'Memory')

        if 'data' in params:
            trace.set_data('data', params['data'])

    def on_trace_start(self, proto, context, options):
        pass

    def on_trace_stop(self, proto, context, options):
        pass

    def on_trace_read(self, proto, context, options):
        if self._framework:
            proto.frameworks.append(self._framework)

    def on_metric_update(self):
        pass