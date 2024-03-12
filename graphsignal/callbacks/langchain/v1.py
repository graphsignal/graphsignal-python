
import logging
import contextvars
from typing import Any, Dict, List, Optional, Union
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish, LLMResult

import graphsignal

logger = logging.getLogger('graphsignal')

span_stack_var = contextvars.ContextVar('langchain_span_stack', default=[])


def push_current_span(span):
    span_stack_var.set(span_stack_var.get() + [span])


def pop_current_span():
    span_stack = span_stack_var.get()
    if len(span_stack) > 0:
        span_stack_var.set(span_stack[:-1])
        return span_stack[-1]
    return None


def get_current_span():
    span_stack = span_stack_var.get()
    if len(span_stack) > 0:
        return span_stack[-1]
    return None


class GraphsignalCallbackHandler(BaseCallbackHandler):
    def __init__(self) -> None:
        self._current_span = None        

    @property
    def always_verbose(self) -> bool:
        return True

    def on_llm_start(
            self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        operation = 'langchain.llms.' + serialized.get('name', 'LLM')
        span = graphsignal.trace(operation)
        push_current_span(span)
        if prompts:
            span.set_payload('prompts', prompts)

    def on_llm_new_token(self, token: str, **kwargs: Any) -> Any:
        pass

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        span = pop_current_span()
        if not span:
            return
        span.stop()

    def on_llm_error(
            self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> None:
        span = pop_current_span()
        if not span:
            return
        if isinstance(error, Exception):
            span.add_exception(error)
        span.stop()

    def on_chain_start(
            self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any) -> None:
        chain_name = serialized.get('name', 'Chain')
        operation = 'langchain.chains.' + chain_name
        span = graphsignal.trace(operation)
        push_current_span(span)
        if inputs:
            span.set_payload('inputs', inputs)

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        span = pop_current_span()
        if not span:
            return
        if outputs:
            span.set_payload('outputs', outputs)
        span.stop()

    def on_chain_error(
            self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> None:
        span = pop_current_span()
        if not span:
            return
        if isinstance(error, Exception):
            span.add_exception(error)
        span.stop()

    def on_tool_start(
            self, serialized: Dict[str, Any], input_str: str, **kwargs: Any) -> None:
        operation = 'langchain.agents.tools.' + serialized.get('name', 'Tool')
        span = graphsignal.trace(operation)
        push_current_span(span)

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        span = pop_current_span()
        if not span:
            return
        if output:
            span.set_payload('output', output)
        span.stop()

    def on_tool_error(
            self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> None:
        span = pop_current_span()
        if not span:
            return
        if isinstance(error, Exception):
            span.add_exception(error)
        span.stop()

    def on_text(self, text: str, **kwargs: Optional[str]) -> None:
        pass

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        pass

    def on_agent_finish(
            self, finish: AgentFinish, **kwargs: Any) -> None:
        pass
