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
        self._framework = None
        self._v1_handler = None

    def setup(self):
        if not graphsignal._agent.auto_instrument:
            return

        self._framework = signals_pb2.FrameworkInfo()
        self._framework.name = 'LangChain'
        version = ''
        if hasattr(langchain, '__version__') and langchain.__version__:
            parse_semver(self._framework.version, langchain.__version__)
            version = langchain.__version__

        if hasattr(langchain.callbacks, 'manager') and hasattr(langchain.callbacks.manager, 'tracing_callback_var'):
            # langchain >= 0.0.154
            from graphsignal.callbacks.langchain.v2 import GraphsignalCallbackHandler
            v2_handler = GraphsignalCallbackHandler()
            langchain.callbacks.manager.tracing_callback_var.set(v2_handler)
            if hasattr(langchain.callbacks.manager, '_configure'):
                def before_configure(args, kwargs):
                    langchain.callbacks.manager.tracing_callback_var.set(v2_handler)
                if not patch_method(langchain.callbacks.manager, '_configure', before_func=before_configure):
                    logger.error(f'Cannot instrument langchain.callbacks.manager._configure for LangChain {version}')
        elif hasattr(langchain.callbacks, 'get_callback_manager'):
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

    def on_trace_read(self, proto, context, options):
        if self._framework:
            proto.frameworks.append(self._framework)
