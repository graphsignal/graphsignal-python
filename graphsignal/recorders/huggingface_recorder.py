import logging
import transformers

import graphsignal
from graphsignal.spans import TraceOptions
from graphsignal.recorders.base_recorder import BaseRecorder
from graphsignal.recorders.instrumentation import instrument_method, uninstrument_method, read_args
from graphsignal.proto_utils import parse_semver, compare_semver
from graphsignal.proto import signals_pb2

logger = logging.getLogger('graphsignal')

class HuggingFaceRecorder(BaseRecorder):
    def __init__(self):
        self._library = None

    def setup(self):
        if not graphsignal._tracer.auto_instrument:
            return

        self._library = signals_pb2.LibraryInfo()
        self._library.name = 'Transformers'
        parse_semver(self._library.version, transformers.__version__)

        from transformers.tools import Agent, RemoteTool, PipelineTool
        instrument_method(Agent, 'run', 'transformers.tools.Agent.run', trace_func=self.trace_run)
        instrument_method(Agent, 'chat', 'transformers.tools.Agent.chat', trace_func=self.trace_run)

        instrument_method(RemoteTool, '__call__', op_func=self.tool_name, trace_func=self.trace_tool)
        instrument_method(PipelineTool, '__call__', op_func=self.tool_name, trace_func=self.trace_tool)

    def shutdown(self):
        from transformers.tools import Agent, RemoteTool, PipelineTool
        uninstrument_method(Agent, 'run')
        uninstrument_method(Agent, 'chat')
        uninstrument_method(RemoteTool, '__call__')
        uninstrument_method(PipelineTool, '__call__')

    def tool_name(self, args, kwargs):
        class_name = args[0].__class__.__name__
        if hasattr(args[0], 'tool_class') and args[0].tool_class:
            class_name = args[0].tool_class.__name__
        else:
            class_name = args[0].__class__.__name__
        return f'transformers.tools.{class_name}'

    def trace_run(self, span, args, kwargs, ret, exc):
        params = read_args(args, kwargs, ['agent', 'task', 'return_code', 'remote'])

        span.set_tag('component', 'Agent')

        if 'return_code' in params:
            span.set_param('return_code', params['return_code'])

        if 'remote' in params:
            span.set_param('remote', params['remote'])

        if 'task' in params:
            span.set_data('task', params['task'])

        if ret:
            span.set_data('output', ret)

    def trace_tool(self, span, args, kwargs, ret, exc):
        span.set_tag('component', 'Tool')
        if hasattr(args[0], 'endpoint_url'):
            span.set_tag('endpoint', args[0].endpoint_url)

        inputs = {}
        if len(args) > 1:
            inputs['args'] = args[1:]
        if len(kwargs) > 0:
            inputs.update(kwargs)
        if len(inputs) > 0:
            span.set_data('inputs', inputs)

        if ret:
            span.set_data('outputs', ret)

    def on_span_read(self, proto, context, options):
        if self._library:
            proto.libraries.append(self._library)
