import logging
from typing import Any, Dict, List, Optional, Union
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish, LLMResult
from uuid import UUID

import graphsignal
from graphsignal.recorders.base_recorder import BaseRecorder
from graphsignal.spans import get_current_span, push_current_span

logger = logging.getLogger('graphsignal')


_trace_graph = {}

class GraphsignalCallbackHandler(BaseCallbackHandler):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def _start_trace(self, parent_run_id, run_id, operation):
        if run_id not in _trace_graph:
            parent_trace = _trace_graph.get(parent_run_id)
            if parent_trace:
                span = get_current_span()
                if not span or span.trace_id != parent_trace._span.trace_id:
                    push_current_span(parent_trace._span)
            trace = graphsignal.start_trace(operation)
            _trace_graph[run_id] = trace
            return trace

    def _current_trace(self, run_id):
        return _trace_graph.get(run_id)

    def _stop_trace(self, run_id, trace):
        trace.stop()
        if run_id in _trace_graph:
            del _trace_graph[run_id]

    def on_llm_start(
            self, 
            serialized: Dict[str, Any], 
            prompts: List[str], 
            *,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            **kwargs: Any) -> None:
        operation = 'langchain.llms.' + serialized.get('name', 'LLM')
        trace = self._start_trace(parent_run_id, run_id, operation)
        if trace:
            trace.set_tag('component', 'LLM')
            if prompts:
                trace.set_data('prompts', prompts)

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
        trace = self._current_trace(run_id)
        if not trace:
            return
        self._stop_trace(run_id, trace)

    def on_llm_error(
            self, 
            error: Union[Exception, KeyboardInterrupt], 
            *,
            run_id: UUID,
            **kwargs: Any) -> None:
        trace = self._current_trace(run_id)
        if not trace:
            return
        if isinstance(error, Exception):
            trace.set_exception(error)
        self._stop_trace(run_id, trace)

    def on_chain_start(
            self, 
            serialized: Dict[str, Any], 
            inputs: Dict[str, Any], 
            *,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            **kwargs: Any) -> None:
        chain_name = serialized.get('name', 'Chain')
        operation = 'langchain.chains.' + chain_name
        trace = self._start_trace(parent_run_id, run_id, operation)
        if trace:
            if chain_name.endswith('AgentExecutor'):
                trace.set_tag('component', 'Agent')
            if inputs:
                trace.set_data('inputs', inputs)

    def on_chain_end(
            self, 
            outputs: Dict[str, Any], 
            *,
            run_id: UUID,
            **kwargs: Any) -> None:
        trace = self._current_trace(run_id)
        if not trace:
            return
        if outputs:
            trace.set_data('outputs', outputs)
        self._stop_trace(run_id, trace)

    def on_chain_error(
            self, 
            error: Union[Exception, KeyboardInterrupt], 
            *,
            run_id: UUID,
            **kwargs: Any) -> None:
        trace = self._current_trace(run_id)
        if not trace:
            return
        if isinstance(error, Exception):
            trace.set_exception(error)
        self._stop_trace(run_id, trace)

    def on_tool_start(
            self, 
            serialized: Dict[str, Any], 
            input_str: str, 
            *,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            **kwargs: Any) -> None:
        operation = 'langchain.agents.tools.' + serialized.get('name', 'Tool')
        trace = self._start_trace(parent_run_id, run_id, operation)
        if trace:
            trace.set_tag('component', 'Tool')

    def on_tool_end(
            self, 
            output: str, 
            *,
            run_id: UUID,
            **kwargs: Any) -> None:
        trace = self._current_trace(run_id)
        if not trace:
            return
        if output:
            trace.set_data('output', output)
        self._stop_trace(run_id, trace)

    def on_tool_error(
            self, 
            error: Union[Exception, KeyboardInterrupt], 
            *,
            run_id: UUID,
            **kwargs: Any) -> None:
        trace = self._current_trace(run_id)
        if not trace:
            return
        if isinstance(error, Exception):
            trace.set_exception(error)
        self._stop_trace(run_id, trace)

    def on_agent_action(
            self,
            action: AgentAction,
            **kwargs: Any) -> None:
        pass

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