import logging
import sys
import time
import contextvars
import langchain
from typing import Any, Dict, List, Optional, Union
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish, LLMResult

import graphsignal
from graphsignal.endpoint_trace import TraceOptions
from graphsignal.recorders.base_recorder import BaseRecorder
from graphsignal.recorders.instrumentation import patch_method, unpatch_method
from graphsignal.proto_utils import parse_semver, compare_semver
from graphsignal.proto import signals_pb2
from graphsignal.proto_utils import add_framework_param, add_driver

logger = logging.getLogger('graphsignal')

trace_stack_var = contextvars.ContextVar('langchain_trace_stack', default=[])


def push_trace(trace):
    trace_stack_var.set(trace_stack_var.get() + [trace])


def pop_trace():
    trace_stack = trace_stack_var.get()
    if len(trace_stack) > 0:
        trace_stack_var.set(trace_stack[:-1])
        return trace_stack[-1]
    return None


class GraphsignalHandler(BaseCallbackHandler):
    def __init__(self) -> None:
        self._current_trace = None        

    @property
    def always_verbose(self) -> bool:
        return True

    def on_llm_start(
            self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        name = 'langchain.llms.' + serialized.get('name', 'LLM')
        trace = graphsignal.start_trace(endpoint=name)
        push_trace(trace)
        if prompts:
            trace.set_data('prompts', prompts)

    def on_llm_new_token(self, token: str, **kwargs: Any) -> Any:
        pass

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        trace = pop_trace()
        if not trace:
            return
        trace.stop()

    def on_llm_error(
            self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> None:
        trace = pop_trace()
        if not trace:
            return
        trace.set_exception(error)
        trace.stop()

    def on_chain_start(
            self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any) -> None:
        name = 'langchain.chains.' + serialized.get('name', 'Chain')
        trace = graphsignal.start_trace(endpoint=name)
        push_trace(trace)
        if inputs:
            trace.set_data('inputs', inputs)

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        trace = pop_trace()
        if not trace:
            return
        if outputs:
            trace.set_data('outputs', outputs)
        trace.stop()

    def on_chain_error(
            self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> None:
        trace = pop_trace()
        if not trace:
            return
        trace.set_exception(error)
        trace.stop()

    def on_tool_start(
            self,
            serialized: Dict[str, Any],
            action: AgentAction,
            color: Optional[str] = None,
            **kwargs: Any) -> None:
        name = 'langchain.agents.tools.' + action.tool
        trace = graphsignal.start_trace(endpoint=name)
        push_trace(trace)

    def on_tool_end(
            self,
            output: str,
            color: Optional[str] = None,
            observation_prefix: Optional[str] = None,
            llm_prefix: Optional[str] = None,
            **kwargs: Any) -> None:
        trace = pop_trace()
        if not trace:
            return
        if output:
            trace.set_data('output', output)
        trace.stop()

    def on_tool_error(
            self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> None:
        trace = pop_trace()
        if not trace:
            return
        trace.set_exception(error)
        trace.stop()

    def on_text(
            self,
            text: str,
            color: Optional[str] = None,
            end: str = "",
            **kwargs: Optional[str]) -> None:
        pass

    def on_agent_finish(
            self, finish: AgentFinish, color: Optional[str] = None, **kwargs: Any) -> None:
        pass


class LangChainRecorder(BaseRecorder):
    def __init__(self):
        self._framework = None
        self._is_sampling = False
        self._handler = None

    def setup(self):
        if not graphsignal._agent.auto_instrument:
            return

        self._framework = signals_pb2.FrameworkInfo()
        self._framework.name = 'LangChain'

        if hasattr(langchain.callbacks, 'get_callback_manager'):
            self._handler = GraphsignalHandler()
            langchain.callbacks.get_callback_manager().add_handler(self._handler)

    def shutdown(self):
        if self._handler:
            langchain.callbacks.get_callback_manager().remove_handler(self._handler)
            self._handler = None

    def on_trace_start(self, signal, context, options):
        self._is_sampling = True

    def on_trace_stop(self, signal, context, options):
        self._is_sampling = False

    def on_trace_read(self, signal, context, options):
        if self._framework:
            signal.frameworks.append(self._framework)
