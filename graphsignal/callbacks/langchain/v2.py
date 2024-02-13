import logging
from typing import Any, Dict, List, Optional, Union, Sequence
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGenerationChunk, GenerationChunk, ChatResult, LLMResult

from uuid import UUID

import graphsignal
from graphsignal.recorders.base_recorder import BaseRecorder
from graphsignal.spans import get_current_span, push_current_span, clear_span_stack

logger = logging.getLogger('graphsignal')

# prevent memory leak, if spans are not stopped for some reason
MAX_TRACES = 10000 

class GraphsignalCallbackHandler(BaseCallbackHandler):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._context_tags = graphsignal._tracer.context_tags.get().copy()
        self._parent_span = get_current_span()

        # is thread-safe, because run_id is unique per thread
        self._span_map = {}
        self._chain_id_map = {}

    def _propagate_context(self, parent_run_id, run_id):
        # propagate context tags just in case contextvars are not propagated
        if self._context_tags:
            for key, value in self._context_tags.items():
                graphsignal.set_context_tag(key, value)

        # propagate chain_id
        chain_id = None
        if not parent_run_id or parent_run_id not in self._chain_id_map:
            # this is the root run, save chain_id
            if run_id:
                chain_id = run_id
                self._chain_id_map[run_id] = chain_id
        else:
            # parent already has a chain_id
            chain_id = self._chain_id_map[parent_run_id]
            self._chain_id_map[run_id] = self._chain_id_map[parent_run_id]

        # set as context tag for other non langchain spans to pick up
        graphsignal.set_context_tag('chain_id', chain_id)

    def _clear_context(self, run_id):
        self._chain_id_map.pop(run_id, None)

    def _get_chain_id(self, run_id):
        return self._chain_id_map.get(run_id, None)

    def _start_trace(self, parent_run_id, run_id, operation):
        if run_id in self._span_map or len(self._span_map) > MAX_TRACES:
            return None

        # set parent span in case it contextvars are not propagated
        if self._parent_span and get_current_span() == None:
            push_current_span(self._parent_span)

        span = graphsignal.start_trace(operation)
        self._span_map[run_id] = span

        chain_id = self._get_chain_id(run_id)
        if chain_id:
            span.set_tag('chain_id', chain_id)

        return span

    def _current_span(self, run_id):
        return self._span_map.get(run_id)

    def _stop_trace(self, run_id):
        span = self._span_map.pop(run_id, None)
        if span:
            span.stop()

    def on_llm_start(
            self, 
            serialized: Dict[str, Any], 
            prompts: List[str], 
            *,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            tags: Optional[List[str]] = None,
            metadata: Optional[Dict[str, Any]] = None,
            **kwargs: Any) -> None:
        try:
            self._propagate_context(parent_run_id, run_id)

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
            tags: Optional[List[str]] = None,
            metadata: Optional[Dict[str, Any]] = None,
            **kwargs: Any) -> Any:
        try:
            self._propagate_context(parent_run_id, run_id)

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
            *,
            chunk: Optional[Union[GenerationChunk, ChatGenerationChunk]] = None,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            tags: Optional[List[str]] = None,
            **kwargs: Any) -> None:
        pass

    def on_llm_end(
            self, 
            response: LLMResult, 
            *, 
            run_id: UUID,
            tags: Optional[List[str]] = None,
            **kwargs: Any) -> None:
        try:
            span = self._current_span(run_id)
            if span:
                if isinstance(response, (LLMResult, ChatResult)):
                    if hasattr(response, 'llm_output'):
                        span.set_data('output', response.llm_output)
                    if hasattr(response, 'generations'):
                        span.set_data('generations', response.generations)
            self._stop_trace(run_id)

            self._clear_context(run_id)
        except Exception:
            logger.error('Error in LangChain callback handler', exc_info=True)

    def on_llm_error(
            self, 
            error: Union[Exception, KeyboardInterrupt], 
            *,
            run_id: UUID,
            tags: Optional[List[str]] = None,
            **kwargs: Any) -> None:
        try:
            span = self._current_span(run_id)
            if span:
                if isinstance(error, Exception):
                    span.add_exception(error)
            self._stop_trace(run_id)

            self._clear_context(run_id)
        except Exception:
            logger.error('Error in LangChain callback handler', exc_info=True)

    def on_chain_start(
            self, 
            serialized: Dict[str, Any], 
            inputs: Dict[str, Any], 
            *,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            tags: Optional[List[str]] = None,
            metadata: Optional[Dict[str, Any]] = None,
            **kwargs: Any) -> None:
        try:
            self._propagate_context(parent_run_id, run_id)
        except Exception:
            logger.error('Error in LangChain callback handler', exc_info=True)

    def on_chain_end(
            self, 
            outputs: Dict[str, Any], 
            *,
            run_id: UUID,
            tags: Optional[List[str]] = None,
            **kwargs: Any) -> None:
        try:
            self._clear_context(run_id)
        except Exception:
            logger.error('Error in LangChain callback handler', exc_info=True)

    def on_chain_error(
            self, 
            error: Union[Exception, KeyboardInterrupt], 
            *,
            run_id: UUID,
            tags: Optional[List[str]] = None,
            **kwargs: Any) -> None:
        try:
            self._clear_context(run_id)
        except Exception:
            logger.error('Error in LangChain callback handler', exc_info=True)

    def on_tool_start(
            self, 
            serialized: Dict[str, Any], 
            input_str: str, 
            *,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            tags: Optional[List[str]] = None,
            metadata: Optional[Dict[str, Any]] = None,
            inputs: Optional[Dict[str, Any]] = None,
            **kwargs: Any) -> None:
        try:
            self._propagate_context(parent_run_id, run_id)

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
            parent_run_id: Optional[UUID] = None,
            tags: Optional[List[str]] = None,
            **kwargs: Any) -> None:
        try:
            span = self._current_span(run_id)
            if span:
                if output:
                    span.set_data('output', output)
            self._stop_trace(run_id)

            self._clear_context(run_id)
        except Exception:
            logger.error('Error in LangChain callback handler', exc_info=True)

    def on_tool_error(
            self, 
            error: Union[Exception, KeyboardInterrupt], 
            *,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            tags: Optional[List[str]] = None,
            **kwargs: Any) -> None:
        try:
            span = self._current_span(run_id)
            if span:
                if isinstance(error, Exception):
                    span.add_exception(error)
            self._stop_trace(run_id)

            self._clear_context(run_id)
        except Exception:
            logger.error('Error in LangChain callback handler', exc_info=True)

    def on_agent_action(
            self,
            action: AgentAction,
            *,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,            
            tags: Optional[List[str]] = None,
            **kwargs: Any) -> None:
        pass

    def on_agent_finish(
            self,
            finish: AgentFinish,
            *,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            tags: Optional[List[str]] = None,
            **kwargs: Any) -> None:
        pass

    def on_text(
            self,
            text: str,
            *,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            tags: Optional[List[str]] = None,
            **kwargs: Any) -> Any:
        pass

    async def on_retriever_start(
            self,
            serialized: Dict[str, Any],
            query: str,
            *,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            tags: Optional[List[str]] = None,
            metadata: Optional[Dict[str, Any]] = None,
            **kwargs: Any) -> None:
        try:
            self._propagate_context(parent_run_id, run_id)

            operation = _get_operation_name(
                serialized=serialized, 
                default_name= 'langchain.retrievers.' + serialized.get('name', 'Retriever'))
            span = self._start_trace(parent_run_id, run_id, operation)
            if span:
                span.set_tag('component', 'Memory')
                if query:
                    span.set_data('query', query)
        except Exception:
            logger.error('Error in LangChain callback handler', exc_info=True)

    async def on_retriever_end(
            self,
            documents: Sequence[Document],
            *,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            tags: Optional[List[str]] = None,
            **kwargs: Any) -> None:
        try:
            span = self._current_span(run_id)
            if span:
                if documents:
                    span.set_data('documents', documents)
            self._stop_trace(run_id)

            self._clear_context(run_id)
        except Exception:
            logger.error('Error in LangChain callback handler', exc_info=True)

    async def on_retriever_error(
            self,
            error: BaseException,
            *,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            tags: Optional[List[str]] = None,
            **kwargs: Any) -> None:
        try:
            span = self._current_span(run_id)
            if span:
                if isinstance(error, Exception):
                    span.add_exception(error)
            self._stop_trace(run_id)

            self._clear_context(run_id)
        except Exception:
            logger.error('Error in LangChain callback handler', exc_info=True)



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
            tags: Optional[List[str]] = None,
            metadata: Optional[Dict[str, Any]] = None,
            **kwargs: Any) -> None:
        super().on_llm_start(
            serialized, 
            prompts, 
            run_id=run_id, 
            parent_run_id=parent_run_id, 
            tags=tags,
            metadata=metadata,
            **kwargs)

    async def on_chat_model_start(
            self,
            serialized: Dict[str, Any],
            messages: List[List[BaseMessage]],
            *,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            tags: Optional[List[str]] = None,
            metadata: Optional[Dict[str, Any]] = None,
            **kwargs: Any) -> Any:
        super().on_chat_model_start(
            serialized, 
            messages, 
            run_id=run_id, 
            parent_run_id=parent_run_id, 
            tags=tags,
            metadata=metadata,
            **kwargs)

    async def on_llm_new_token(
            self,
            token: str,
            *,
            chunk: Optional[Union[GenerationChunk, ChatGenerationChunk]] = None,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            tags: Optional[List[str]] = None,
            **kwargs: Any) -> None:
        pass

    async def on_llm_end(
            self, 
            response: LLMResult, 
            *, 
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            tags: Optional[List[str]] = None,
            **kwargs: Any) -> None:
        super().on_llm_end(
            response,
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
            **kwargs)

    async def on_llm_error(
            self, 
            error: Union[Exception, KeyboardInterrupt], 
            *,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            tags: Optional[List[str]] = None,
            **kwargs: Any) -> None:
        super().on_llm_error(
            error,
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
            **kwargs)

    async def on_chain_start(
            self, 
            serialized: Dict[str, Any], 
            inputs: Dict[str, Any], 
            *,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            tags: Optional[List[str]] = None,
            **kwargs: Any) -> None:
        super().on_chain_start(
            serialized,
            inputs,
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
            **kwargs)

    async def on_chain_end(
            self, 
            outputs: Dict[str, Any], 
            *,
            run_id: UUID,
            tags: Optional[List[str]] = None,
            **kwargs: Any) -> None:
        super().on_chain_end(
            outputs,
            run_id=run_id,
            tags=tags,
            **kwargs)

    async def on_chain_error(
            self, 
            error: Union[Exception, KeyboardInterrupt], 
            *,
            run_id: UUID,
            tags: Optional[List[str]] = None,
            **kwargs: Any) -> None:
        super().on_chain_error(
            error,
            run_id=run_id,
            tags=tags,
            **kwargs)

    async def on_tool_start(
            self, 
            serialized: Dict[str, Any], 
            input_str: str, 
            *,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            tags: Optional[List[str]] = None,
            metadata: Optional[Dict[str, Any]] = None,
            inputs: Optional[Dict[str, Any]] = None,
            **kwargs: Any) -> None:
        super().on_tool_start(
            serialized,
            input_str,
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
            metadata=metadata,
            inputs=inputs,
            **kwargs)

    async def on_tool_end(
            self, 
            output: str, 
            *,
            run_id: UUID,
            tags: Optional[List[str]] = None,
            **kwargs: Any) -> None:
        super().on_tool_end(
            output,
            run_id=run_id,
            tags=tags,
            **kwargs)

    async def on_tool_error(
            self, 
            error: Union[Exception, KeyboardInterrupt], 
            *,
            run_id: UUID,
            tags: Optional[List[str]] = None,
            **kwargs: Any) -> None:
        super().on_tool_error(
            error,
            run_id=run_id,
            tags=tags,
            **kwargs)

    async def on_retriever_start(
            self,
            serialized: Dict[str, Any],
            query: str,
            *,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            tags: Optional[List[str]] = None,
            metadata: Optional[Dict[str, Any]] = None,
            **kwargs: Any) -> None:
        super().on_retriever_start(
            serialized,
            query,
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
            metadata=metadata,
            **kwargs)

    async def on_retriever_end(
            self,
            documents: Sequence[Document],
            *,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            tags: Optional[List[str]] = None,
            **kwargs: Any) -> None:
        super().on_retriever_end(
            documents,
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
            **kwargs)

    async def on_retriever_error(
            self,
            error: BaseException,
            *,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            tags: Optional[List[str]] = None,
            **kwargs: Any) -> None:
        super().on_retriever_error(
            error,
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
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
