import logging
from typing import Any, Dict, List, Optional, Union
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish, LLMResult, ChatResult, Generation, ChatGeneration
from uuid import UUID

import graphsignal
from graphsignal.recorders.base_recorder import BaseRecorder
from graphsignal.spans import get_current_span, push_current_span, clear_span_stack
from graphsignal.data.utils import obj_to_dict

logger = logging.getLogger('graphsignal')

# prevent memory leak, if traces are not stopped for some reason
MAX_TRACES = 10000 

# is thread-safe, because run_id is unique per thread
_trace_graph = {}


class GraphsignalCallbackHandler(BaseCallbackHandler):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._context_tags = graphsignal._agent.context_tags.get().copy()
        self._parent_span = get_current_span()

    def _start_trace(self, parent_run_id, run_id, operation):
        if run_id in _trace_graph or len(_trace_graph) > MAX_TRACES:
            return None

        # do not rely on contextvars in callbacks
        clear_span_stack()

        # set parent span
        if not parent_run_id:
            # initialize handler
            if self._parent_span:
                push_current_span(self._parent_span)
        else:
            parent_trace = _trace_graph.get(parent_run_id)
            if parent_trace:
                push_current_span(parent_trace._span)
            else:
                return None

        # set context tags
        if self._context_tags:
            for key, value in self._context_tags.items():
                graphsignal.set_context_tag(key, value)

        trace = graphsignal.start_trace(operation)
        _trace_graph[run_id] = trace
        return trace

    def _current_trace(self, run_id):
        return _trace_graph.get(run_id)

    def _stop_trace(self, run_id):
        trace = _trace_graph.pop(run_id, None)
        if trace:
            trace.stop()

    def on_llm_start(
            self, 
            serialized: Dict[str, Any], 
            prompts: List[str], 
            *,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            **kwargs: Any) -> None:
        try:
            operation = 'langchain.llms.' + serialized.get('name', 'LLM')
            trace = self._start_trace(parent_run_id, run_id, operation)
            if trace:
                trace.set_tag('component', 'LLM')
                if prompts:
                    trace.set_data('prompts', prompts)
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
            trace = self._current_trace(run_id)
            if trace:
                if isinstance(response, (LLMResult, ChatResult)) and hasattr(response, 'llm_output'):
                    trace.set_data('output', response.llm_output)
                    trace.set_data('generations', obj_to_dict(response.generations))
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
            trace = self._current_trace(run_id)
            if trace:
                if isinstance(error, Exception):
                    trace.set_exception(error)
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
            chain_name = serialized.get('name', 'Chain')
            operation = 'langchain.chains.' + chain_name
            trace = self._start_trace(parent_run_id, run_id, operation)
            if trace:
                trace.set_tag('component', 'Agent')
                if inputs:
                    trace.set_data('inputs', inputs)
        except Exception:
            logger.error('Error in LangChain callback handler', exc_info=True)

    def on_chain_end(
            self, 
            outputs: Dict[str, Any], 
            *,
            run_id: UUID,
            **kwargs: Any) -> None:
        try:
            trace = self._current_trace(run_id)
            if trace:
                if outputs:
                    trace.set_data('outputs', outputs)
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
            trace = self._current_trace(run_id)
            if trace:
                if isinstance(error, Exception):
                    trace.set_exception(error)
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
            operation = 'langchain.agents.tools.' + serialized.get('name', 'Tool')
            trace = self._start_trace(parent_run_id, run_id, operation)
            if trace:
                trace.set_tag('component', 'Tool')
                if input_str:
                    trace.set_data('input', input_str)
        except Exception:
            logger.error('Error in LangChain callback handler', exc_info=True)

    def on_tool_end(
            self, 
            output: str, 
            *,
            run_id: UUID,
            **kwargs: Any) -> None:
        try:
            trace = self._current_trace(run_id)
            if trace:
                if output:
                    trace.set_data('output', output)
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
            trace = self._current_trace(run_id)
            if trace:
                if isinstance(error, Exception):
                    trace.set_exception(error)
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
            trace = self._current_trace(run_id)
            if trace:
                if isinstance(action, AgentAction):
                    trace.append_data('actions', [obj_to_dict(action)])
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
