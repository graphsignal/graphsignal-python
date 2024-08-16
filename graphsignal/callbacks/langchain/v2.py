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
from graphsignal.utils import uuid_sha1

logger = logging.getLogger('graphsignal')

# prevent memory leak, if spans are not stopped for some reason
MAX_TRACES = 10000 

class GraphsignalCallbackHandler(BaseCallbackHandler):
    def __init__(self, **kwargs: Any) -> None:
        self._passed_tags = kwargs.pop('tags', None) or {}
        super().__init__(**kwargs)
        self._context_tags = graphsignal._tracer.context_tags.get().copy()

        # is thread-safe, because run_id is unique per thread
        self._span_map = {}

        # create session_id, if not passed
        self._generated_session_id = False
        if 'session_id' not in self._passed_tags and 'session_id' not in self._context_tags:
            self._passed_tags['session_id'] = uuid_sha1(size=12)
            self._generated_session_id = True

    def _start_trace(self, parent_run_id, run_id, operation):
        # propagate context tags just in case contextvars are not propagated
        if self._passed_tags:
            for key, value in self._passed_tags.items():
                graphsignal.set_context_tag(key, value)
        if self._context_tags:
            for key, value in self._context_tags.items():
                graphsignal.set_context_tag(key, value)

        if run_id in self._span_map or len(self._span_map) > MAX_TRACES:
            return None

        span = graphsignal.trace(operation, tags=self._passed_tags)
        self._span_map[run_id] = span

        span.set_tag('library', 'langchain')

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
            operation = _get_operation_name(
                serialized=serialized, 
                default_name='langchain.llms.' + serialized.get('name', 'LLM'))
            span = self._start_trace(parent_run_id, run_id, operation)
            if span:
                span.set_tag('model_type', 'chat')
                input = dict(messages=[])
                if prompts:
                    for prompt in prompts:
                        input['messages'].append(dict(content=prompt))
                if len(input['messages']) > 0:
                    span.set_payload('input', input)
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
            operation = _get_operation_name(
                serialized=serialized, 
                default_name='langchain.llms.' + serialized.get('name', 'LLM'))
            span = self._start_trace(parent_run_id, run_id, operation)
            if span:
                span.set_tag('model_type', 'chat')
                input = dict(messages=[])
                if messages:
                    for message_list in messages:
                        for message in message_list:
                            if hasattr(message, 'content'):
                                input['messages'].append(dict(content=message.content))
                if len(input['messages']) > 0:
                    span.set_payload('input', input)
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
                output = dict(choices=[])
                if isinstance(response, ChatResult):
                    if hasattr(response, 'generations') and isinstance(response.generations, list):
                        for generation in response.generations:
                            if hasattr(generation, 'message'):
                                if hasattr(generation.message, 'content'):
                                    output['choices'].append(dict(message=dict(content=generation.message.content)))
                            elif hasattr(generation, 'text'):
                                output['choices'].append(dict(message=dict(content=generation.text)))
                elif isinstance(response, LLMResult):
                    if hasattr(response, 'generations') and isinstance(response.generations, list):
                        for generation_list in response.generations:
                            for generation in generation_list:
                                if hasattr(generation, 'message'):
                                    if hasattr(generation.message, 'content'):
                                        output['choices'].append(dict(message=dict(content=generation.message.content)))
                                elif hasattr(generation, 'text'):
                                    output['choices'].append(dict(message=dict(content=generation.text)))
                if len(output['choices']) > 0:
                    span.set_payload('output', output)
            self._stop_trace(run_id)
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
            operation = _get_operation_name(
                serialized=serialized, 
                default_name= 'langchain.agents.agent.' + serialized.get('name', 'AgentExecutor'))
            span = self._start_trace(parent_run_id, run_id, operation)
            if span:
                if inputs:
                    span.set_payload('inputs', inputs)
                if metadata:
                    span.set_payload('metadata', metadata)
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
            span = self._current_span(run_id)
            if span:
                if outputs:
                    span.set_payload('outputs', outputs)

            self._stop_trace(run_id)
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
            tags: Optional[List[str]] = None,
            metadata: Optional[Dict[str, Any]] = None,
            inputs: Optional[Dict[str, Any]] = None,
            **kwargs: Any) -> None:
        try:
            operation = _get_operation_name(
                serialized=serialized, 
                default_name= 'langchain.agents.tools.' + serialized.get('name', 'Tool'))
            span = self._start_trace(parent_run_id, run_id, operation)
            if span:
                if input_str:
                    span.set_payload('input', input_str)
                if metadata:
                    span.set_payload('metadata', metadata)
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
                    span.set_payload('output', output)
            self._stop_trace(run_id)
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
        try:
            span = self._current_span(run_id)
            if span:
                span.set_payload('action', dict(tool=action.tool, tool_input=action.tool_input))
        except Exception:
            logger.error('Error in LangChain callback handler', exc_info=True)

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
            operation = _get_operation_name(
                serialized=serialized, 
                default_name= 'langchain.retrievers.' + serialized.get('name', 'Retriever'))
            span = self._start_trace(parent_run_id, run_id, operation)
            if span:
                if query:
                    span.set_payload('query', query)
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
                    span.set_payload('documents', documents)
            self._stop_trace(run_id)
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
        if self._generated_session_id:
            graphsignal.set_context_tag('session_id', None)

    async def on_text(
            self,
            text: str,
            **kwargs: Any) -> Any:
        pass
