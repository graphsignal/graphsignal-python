import logging
import transformers

import graphsignal
from graphsignal.spans import TraceOptions
from graphsignal.recorders.base_recorder import BaseRecorder
from graphsignal.recorders.instrumentation import instrument_method, uninstrument_method, read_args
from graphsignal.proto_utils import parse_semver, compare_semver
from graphsignal.proto import signals_pb2
from graphsignal.proto_utils import add_library_param, add_driver

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
        instrument_method(Agent, 'run', 'transformer.tools.Agent.run', self.trace_run)
        instrument_method(Agent, 'chat', 'transformer.tools.Agent.chat', self.trace_run)

        instrument_method(RemoteTool, '__call__', 'transformer.tools.RemoteTool', self.trace_tool)
        instrument_method(PipelineTool, '__call__', 'transformer.tools.PipelineTool', self.trace_tool)

    def shutdown(self):
        from transformers.tools import Agent, RemoteTool, PipelineTool
        uninstrument_method(Agent, 'run', 'transformer.tools.Agent.run')
        uninstrument_method(Agent, 'chat', 'transformer.tools.Agent.chat')
        uninstrument_method(RemoteTool, '__call__', 'transformer.tools.RemoteTool')
        uninstrument_method(PipelineTool, '__call__', 'transformer.tools.PipelineTool')

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
        params = read_args(args, kwargs, ['agent', 'task', 'return_code', 'remote'])

        span.set_tag('component', 'Tool')

        if ret:
            span.set_data('outputs', ret)

    def on_span_read(self, proto, context, options):
        if self._library:
            proto.libraries.append(self._library)
