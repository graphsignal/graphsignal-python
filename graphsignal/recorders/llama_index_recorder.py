import logging
import llama_index

import graphsignal
from graphsignal.recorders.base_recorder import BaseRecorder
from graphsignal.recorders.instrumentation import parse_semver, compare_semver
from graphsignal.proto import signals_pb2
from graphsignal.recorders.instrumentation import patch_method

logger = logging.getLogger('graphsignal')


class LlamaIndexRecorder(BaseRecorder):
    def __init__(self):
        self._library_version = None
        self._v1_handler = None

    def setup(self):
        if not graphsignal._tracer.auto_instrument:
            return

        version = ''
        if hasattr(llama_index, '__version__') and llama_index.__version__:
            version = llama_index.__version__
            self._library_version = version

        def is_v1():
            return (
                hasattr(llama_index, 'indices') and hasattr(llama_index.indices.service_context, 'ServiceContext')
            )

        def is_v2():
            return self._library_version and compare_semver(self._library_version, (0, 10, 10)) >= 0

        if is_v2():
            # the handler should be added manually for now
            pass
        elif is_v1():
            from graphsignal.callbacks.llama_index.v1 import GraphsignalCallbackHandler
            from llama_index.indices.service_context import ServiceContext
            def after_from_defaults(args, kwargs, ret, exc, context):
                if isinstance(ret, ServiceContext):
                    if not any(isinstance(handler, GraphsignalCallbackHandler) for handler in ret.callback_manager.handlers):
                        ret.callback_manager.add_handler(GraphsignalCallbackHandler())
                else:
                    logger.error(f'Cannot add callback for LlamaIndex {version}')
            if not patch_method(ServiceContext, 'from_defaults', after_func=after_from_defaults):
                logger.error(f'Cannot instrument LlamaIndex {version}')
        else:
            logger.error(f'Cannot auto-instrument LlamaIndex {version}')

    def shutdown(self):
        if self._v1_handler:
            llama_index.callbacks.get_callback_manager().remove_handler(self._v1_handler)
            self._v1_handler = None

    def on_span_read(self, span, context):
        if self._library_version:
            entry = span.config.add()
            entry.key = 'llama_index'
            entry.value = self._library_version
