import logging
from typing import Any, Dict, List, Optional, Union
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish, LLMResult, ChatResult, BaseMessage
from uuid import UUID

import graphsignal
from graphsignal.recorders.base_recorder import BaseRecorder
from graphsignal.spans import get_current_span, push_current_span, clear_span_stack

logger = logging.getLogger('graphsignal')

# prevent memory leak, if spans are not stopped for some reason
MAX_TRACES = 10000 

# is thread-safe, because run_id is unique per thread
_span_map = {}


class GraphsignalCallbackHandler(BaseCallbackHandler):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._context_tags = graphsignal._tracer.context_tags.get().copy()
        self._parent_span = get_current_span()

    def _start_trace(self, parent_run_id, run_id, operation):
        if run_id in _span_map or len(_span_map) > MAX_TRACES:
            return None

        # do not rely on contextvars in callbacks
        clear_span_stack()

        # set parent span
        if not parent_run_id:
            # initialize handler
            if self._parent_span:
                push_current_span(self._parent_span)
        else:
            parent_span = _span_map.get(parent_run_id)
            if parent_span:
                push_current_span(parent_span)
            else:
                return None

        # set context tags
        if self._context_tags:
            for key, value in self._context_tags.items():
                graphsignal.set_context_tag(key, value)

        span = graphsignal.start_trace(operation)
        _span_map[run_id] = span
        return span

    def _current_span(self, run_id):
        return _span_map.get(run_id)

    def _stop_trace(self, run_id):
        span = _span_map.pop(run_id, None)
        if span:
            span.stop()

    def on_llm_start(
            self, 
            serialized: Dict[str, Any], 
            prompts: List[str], 
            *,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            **kwargs: Any) -> None:
        try:
            operation = _get_operation_name(
                serialized=serialized, 
                default_name='langchain.llms.' + serialized.get('name', 'LLM'))
            span = self._start_trace(parent_run_id, run_id, operation)
            if span:
                span.set_tag('component', 'LLM')
                if prompts:
                    span.set_data('prompts', prompts)
        except Exception:
            logger.error('Error in LangChain callback handler', exc_info=True)

    def on_chat_model_start(
            self,
            serialized: Dict[str, Any],
            messages: List[List[BaseMessage]],
            *,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            **kwargs: Any) -> Any:
        try:
            operation = _get_operation_name(
                serialized=serialized, 
                default_name='langchain.llms.' + serialized.get('name', 'LLM'))
            span = self._start_trace(parent_run_id, run_id, operation)
            if span:
                span.set_tag('component', 'LLM')
                if messages:
                    span.set_data('messages', messages)
        except Exception:
            logger.error('Error in LangChain callback handler', exc_info=True)

    
    def on_llm_new_token(
            self,
            token: str,
            **kwargs: Any) -> None:
        pass

    def on_llm_end(
            self, 
            response: LLMResult, 
            *, 
            run_id: UUID,
            **kwargs: Any) -> None:
        try:
            span = self._current_span(run_id)
            if span:
                if isinstance(response, (LLMResult, ChatResult)) and hasattr(response, 'llm_output'):
                    span.set_data('output', response.llm_output)
                    span.set_data('generations', response.generations)
            self._stop_trace(run_id)
        except Exception:
            logger.error('Error in LangChain callback handler', exc_info=True)

    def on_llm_error(
            self, 
            error: Union[Exception, KeyboardInterrupt], 
            *,
            run_id: UUID,
            **kwargs: Any) -> None:
        try:
            span = self._current_span(run_id)
            if span:
                if isinstance(error, Exception):
                    span.add_exception(error)
            self._stop_trace(run_id)
        except Exception:
            logger.error('Error in LangChain callback handler', exc_info=True)

    def on_chain_start(
            self, 
            serialized: Dict[str, Any], 
            inputs: Dict[str, Any], 
            *,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            **kwargs: Any) -> None:
        try:
            operation = _get_operation_name(
                serialized=serialized, 
                default_name='langchain.chains.' + serialized.get('name', 'Chain'))
            span = self._start_trace(parent_run_id, run_id, operation)
            if span:
                span.set_tag('component', 'Agent')
                if inputs:
                    span.set_data('inputs', inputs)
        except Exception:
            logger.error('Error in LangChain callback handler', exc_info=True)

    def on_chain_end(
            self, 
            outputs: Dict[str, Any], 
            *,
            run_id: UUID,
            **kwargs: Any) -> None:
        try:
            span = self._current_span(run_id)
            if span:
                if outputs:
                    span.set_data('outputs', outputs)
            self._stop_trace(run_id)
        except Exception:
            logger.error('Error in LangChain callback handler', exc_info=True)

    def on_chain_error(
            self, 
            error: Union[Exception, KeyboardInterrupt], 
            *,
            run_id: UUID,
            **kwargs: Any) -> None:
        try:
            span = self._current_span(run_id)
            if span:
                if isinstance(error, Exception):
                    span.add_exception(error)
            self._stop_trace(run_id)
        except Exception:
            logger.error('Error in LangChain callback handler', exc_info=True)

    def on_tool_start(
            self, 
            serialized: Dict[str, Any], 
            input_str: str, 
            *,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            **kwargs: Any) -> None:
        try:
            operation = _get_operation_name(
                serialized=serialized, 
                default_name= 'langchain.agents.tools.' + serialized.get('name', 'Tool'))
            span = self._start_trace(parent_run_id, run_id, operation)
            if span:
                span.set_tag('component', 'Tool')
                if input_str:
                    span.set_data('input', input_str)
        except Exception:
            logger.error('Error in LangChain callback handler', exc_info=True)

    def on_tool_end(
            self, 
            output: str, 
            *,
            run_id: UUID,
            **kwargs: Any) -> None:
        try:
            span = self._current_span(run_id)
            if span:
                if output:
                    span.set_data('output', output)
            self._stop_trace(run_id)
        except Exception:
            logger.error('Error in LangChain callback handler', exc_info=True)

    def on_tool_error(
            self, 
            error: Union[Exception, KeyboardInterrupt], 
            *,
            run_id: UUID,
            **kwargs: Any) -> None:
        try:
            span = self._current_span(run_id)
            if span:
                if isinstance(error, Exception):
                    span.add_exception(error)
            self._stop_trace(run_id)
        except Exception:
            logger.error('Error in LangChain callback handler', exc_info=True)

    def on_agent_action(
            self,
            action: AgentAction,
            *,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,            
            **kwargs: Any) -> None:
        try:
            span = self._current_span(run_id)
            if span:
                if isinstance(action, AgentAction):
                    span.append_data('actions', [action])
        except Exception:
            logger.error('Error in LangChain callback handler', exc_info=True)

    def on_agent_finish(
            self,
            finish: AgentFinish,
            **kwargs: Any) -> None:
        pass

    def on_text(
            self,
            text: str,
            **kwargs: Any) -> Any:
        pass


def _get_operation_name(serialized, default_name):
    if 'id' in serialized and isinstance(serialized['id'], list):
        return '.'.join(serialized['id'])
    return default_name

class GraphsignalAsyncCallbackHandler(GraphsignalCallbackHandler):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    async def on_llm_start(
            self, 
            serialized: Dict[str, Any], 
            prompts: List[str], 
            *,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            **kwargs: Any) -> None:
        super().on_llm_start(
            serialized, 
            prompts, 
            run_id=run_id, 
            parent_run_id=parent_run_id, 
            **kwargs)

    async def on_chat_model_start(
            self,
            serialized: Dict[str, Any],
            messages: List[List[BaseMessage]],
            *,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            **kwargs: Any) -> Any:
        super().on_chat_model_start(
            serialized, 
            messages, 
            run_id=run_id, 
            parent_run_id=parent_run_id, 
            **kwargs)

    async def on_llm_new_token(
            self,
            token: str,
            **kwargs: Any) -> None:
        pass

    async def on_llm_end(
            self, 
            response: LLMResult, 
            *, 
            run_id: UUID,
            **kwargs: Any) -> None:
        super().on_llm_end(
            response,
            run_id=run_id,
            **kwargs)

    async def on_llm_error(
            self, 
            error: Union[Exception, KeyboardInterrupt], 
            *,
            run_id: UUID,
            **kwargs: Any) -> None:
        super().on_llm_error(
            error,
            run_id=run_id,
            **kwargs)

    async def on_chain_start(
            self, 
            serialized: Dict[str, Any], 
            inputs: Dict[str, Any], 
            *,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            **kwargs: Any) -> None:
        super().on_chain_start(
            serialized,
            inputs,
            run_id=run_id,
            parent_run_id=parent_run_id,
            **kwargs)

    async def on_chain_end(
            self, 
            outputs: Dict[str, Any], 
            *,
            run_id: UUID,
            **kwargs: Any) -> None:
        super().on_chain_end(
            outputs,
            run_id=run_id,
            **kwargs)

    async def on_chain_error(
            self, 
            error: Union[Exception, KeyboardInterrupt], 
            *,
            run_id: UUID,
            **kwargs: Any) -> None:
        super().on_chain_error(
            error,
            run_id=run_id,
            **kwargs)

    async def on_tool_start(
            self, 
            serialized: Dict[str, Any], 
            input_str: str, 
            *,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            **kwargs: Any) -> None:
        super().on_tool_start(
            serialized,
            input_str,
            run_id=run_id,
            parent_run_id=parent_run_id,
            **kwargs)

    async def on_tool_end(
            self, 
            output: str, 
            *,
            run_id: UUID,
            **kwargs: Any) -> None:
        super().on_tool_end(
            output,
            run_id=run_id,
            **kwargs)

    async def on_tool_error(
            self, 
            error: Union[Exception, KeyboardInterrupt], 
            *,
            run_id: UUID,
            **kwargs: Any) -> None:
        super().on_tool_error(
            error,
            run_id=run_id,
            **kwargs)

    async def on_agent_action(
            self,
            action: AgentAction,
            *,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,            
            **kwargs: Any) -> None:
        pass

    async def on_agent_finish(
            self,
            finish: AgentFinish,
            **kwargs: Any) -> None:
        pass

    async def on_text(
            self,
            text: str,
            **kwargs: Any) -> Any:
        pass
