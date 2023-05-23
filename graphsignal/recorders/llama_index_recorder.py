import logging
import llama_index

import graphsignal
from graphsignal.recorders.base_recorder import BaseRecorder
from graphsignal.proto_utils import parse_semver
from graphsignal.proto import signals_pb2
from graphsignal.recorders.instrumentation import patch_method

logger = logging.getLogger('graphsignal')


class LlamaIndexRecorder(BaseRecorder):
    def __init__(self):
        self._library = None
        self._v1_handler = None

    def setup(self):
        if not graphsignal._tracer.auto_instrument:
            return

        self._library = signals_pb2.LibraryInfo()
        self._library.name = 'LlamaIndex'
        version = ''
        if hasattr(llama_index, '__version__') and llama_index.__version__:
            parse_semver(self._library.version, llama_index.__version__)
            version = llama_index.__version__

        def is_v1():
            return (
                hasattr(llama_index.indices.service_context, 'ServiceContext')
            )

        if is_v1():
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

    def on_span_read(self, proto, context, options):
        if self._library:
            proto.libraries.append(self._library)
