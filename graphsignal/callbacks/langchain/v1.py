
import logging
import contextvars
from typing import Any, Dict, List, Optional, Union
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish, LLMResult

import graphsignal

logger = logging.getLogger('graphsignal')

trace_stack_var = contextvars.ContextVar('langchain_trace_stack', default=[])


def push_current_trace(trace):
    trace_stack_var.set(trace_stack_var.get() + [trace])


def pop_current_trace():
    trace_stack = trace_stack_var.get()
    if len(trace_stack) > 0:
        trace_stack_var.set(trace_stack[:-1])
        return trace_stack[-1]
    return None


def get_current_trace():
    trace_stack = trace_stack_var.get()
    if len(trace_stack) > 0:
        return trace_stack[-1]
    return None


class GraphsignalCallbackHandler(BaseCallbackHandler):
    def __init__(self) -> None:
        self._current_trace = None        

    @property
    def always_verbose(self) -> bool:
        return True

    def on_llm_start(
            self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        operation = 'langchain.llms.' + serialized.get('name', 'LLM')
        trace = graphsignal.start_trace(operation)
        trace.set_tag('component', 'LLM')
        push_current_trace(trace)
        if prompts:
            trace.set_data('prompts', prompts)

    def on_llm_new_token(self, token: str, **kwargs: Any) -> Any:
        pass

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        trace = pop_current_trace()
        if not trace:
            return
        trace.stop()

    def on_llm_error(
            self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> None:
        trace = pop_current_trace()
        if not trace:
            return
        if isinstance(error, Exception):
            trace.set_exception(error)
        trace.stop()

    def on_chain_start(
            self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any) -> None:
        chain_name = serialized.get('name', 'Chain')
        operation = 'langchain.chains.' + chain_name
        trace = graphsignal.start_trace(operation)
        if chain_name.endswith('AgentExecutor'):
            trace.set_tag('component', 'Agent')
        push_current_trace(trace)
        if inputs:
            trace.set_data('inputs', inputs)

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        trace = pop_current_trace()
        if not trace:
            return
        if outputs:
            trace.set_data('outputs', outputs)
        trace.stop()

    def on_chain_error(
            self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> None:
        trace = pop_current_trace()
        if not trace:
            return
        if isinstance(error, Exception):
            trace.set_exception(error)
        trace.stop()

    def on_tool_start(
            self, serialized: Dict[str, Any], input_str: str, **kwargs: Any) -> None:
        operation = 'langchain.agents.tools.' + serialized.get('name', 'Tool')
        trace = graphsignal.start_trace(operation)
        trace.set_tag('component', 'Tool')
        push_current_trace(trace)

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        trace = pop_current_trace()
        if not trace:
            return
        if output:
            trace.set_data('output', output)
        trace.stop()

    def on_tool_error(
            self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> None:
        trace = pop_current_trace()
        if not trace:
            return
        if isinstance(error, Exception):
            trace.set_exception(error)
        trace.stop()

    def on_text(self, text: str, **kwargs: Optional[str]) -> None:
        pass

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        pass

    def on_agent_finish(
            self, finish: AgentFinish, **kwargs: Any) -> None:
        pass
