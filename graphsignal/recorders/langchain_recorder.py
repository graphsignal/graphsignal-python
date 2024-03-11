import logging
import langchain

import graphsignal
from graphsignal.recorders.base_recorder import BaseRecorder
from graphsignal.recorders.instrumentation import patch_method

logger = logging.getLogger('graphsignal')


class LangChainRecorder(BaseRecorder):
    def __init__(self):
        self._library_version = None
        self._v1_handler = None

    def setup(self):
        if not graphsignal._tracer.auto_instrument:
            return

        version = ''
        if hasattr(langchain, '__version__') and langchain.__version__:
            version = langchain.__version__
            self._library_version = version

        def is_v2():
            try:
                from langchain.callbacks.manager import CallbackManager
                from langchain.callbacks.manager import AsyncCallbackManager
            except ImportError:
                return False
            return True

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

    def on_span_read(self, span, context):
        if self._library_version:
            entry = span.config.add()
            entry.key = 'langchain.library.version'
            entry.value = self._library_version
