import logging
import langchain

import graphsignal
from graphsignal.recorders.base_recorder import BaseRecorder
from graphsignal.proto_utils import parse_semver
from graphsignal.proto import signals_pb2
from graphsignal.recorders.instrumentation import patch_method

logger = logging.getLogger('graphsignal')


class LangChainRecorder(BaseRecorder):
    def __init__(self):
        self._library = None
        self._v1_handler = None

    def setup(self):
        if not graphsignal._tracer.auto_instrument:
            return

        self._library = signals_pb2.LibraryInfo()
        self._library.name = 'LangChain'
        version = ''
        if hasattr(langchain, '__version__') and langchain.__version__:
            parse_semver(self._library.version, langchain.__version__)
            version = langchain.__version__

        def is_v2():
            return (
                hasattr(langchain.callbacks, 'manager') and
                hasattr(langchain.callbacks.manager, 'CallbackManager') and
                hasattr(langchain.callbacks.manager, 'AsyncCallbackManager')
            )

        def is_v1():
            return hasattr(langchain.callbacks, 'get_callback_manager')


        if is_v2():
            # langchain >= 0.0.154
            from graphsignal.callbacks.langchain.v2 import GraphsignalCallbackHandler
            def after_configure(args, kwargs, ret, exc, context):
                if isinstance(ret, langchain.callbacks.manager.CallbackManager):
                    if not any(isinstance(handler, GraphsignalCallbackHandler) for handler in ret.handlers):
                        ret.add_handler(GraphsignalCallbackHandler())
                else:
                    logger.error(f'Cannot add callback for LangChain {version}')
            if not patch_method(langchain.callbacks.manager.CallbackManager, 'configure', after_func=after_configure):
                logger.error(f'Cannot instrument LangChain {version}')

            from graphsignal.callbacks.langchain.v2 import GraphsignalAsyncCallbackHandler
            def after_async_configure(args, kwargs, ret, exc, context):
                if isinstance(ret, langchain.callbacks.manager.AsyncCallbackManager):
                    if not any(isinstance(handler, GraphsignalAsyncCallbackHandler) for handler in ret.handlers):
                        ret.add_handler(GraphsignalAsyncCallbackHandler())
                else:
                    logger.error(f'Cannot add callback for LangChain {version}')
            if not patch_method(langchain.callbacks.manager.AsyncCallbackManager, 'configure', after_func=after_async_configure):
                logger.error(f'Cannot instrument LangChain {version}')

        elif is_v1():
            # compatibility with langchain <= 0.0.153
            from graphsignal.callbacks.langchain.v1 import GraphsignalCallbackHandler
            self._v1_handler = GraphsignalCallbackHandler()
            langchain.callbacks.get_callback_manager().add_handler(self._v1_handler)

        else:
            logger.error(f'Cannot auto-instrument LangChain {version}')

    def shutdown(self):
        if self._v1_handler:
            langchain.callbacks.get_callback_manager().remove_handler(self._v1_handler)
            self._v1_handler = None

    def on_span_read(self, proto, context, options):
        if self._library:
            proto.libraries.append(self._library)
