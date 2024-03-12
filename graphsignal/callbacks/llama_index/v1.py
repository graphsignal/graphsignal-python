import logging
from typing import Any, Dict, List, Optional, Union
from uuid import UUID
import contextvars
from llama_index.callbacks.base import BaseCallbackHandler
from llama_index.callbacks.schema import CBEventType

import graphsignal
from graphsignal.recorders.base_recorder import BaseRecorder
from graphsignal.spans import get_current_span, push_current_span, clear_span_stack

logger = logging.getLogger('graphsignal')

# prevent memory leak, if spans are not stopped for some reason
MAX_TRACES = 10000 

# is thread-safe, because event_id is unique per thread
_span_map = {}

_llama_trace_map = contextvars.ContextVar('llama_trace_map', default={})


class GraphsignalCallbackHandler(BaseCallbackHandler):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            event_starts_to_ignore=[],
            event_ends_to_ignore=[]
        )

    def _start_trace(self, event_id, operation):
        if event_id in _span_map or len(_span_map) > MAX_TRACES:
            return None

        span = graphsignal.trace(operation)
        _span_map[event_id] = span
        return span

    def _current_span(self, event_id):
        return _span_map.get(event_id)

    def _stop_trace(self, event_id):
        span = _span_map.pop(event_id, None)
        if span:
            span.stop()

    def on_event_start(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any
    ) -> str:
        try:
            event_name_str = event_type.name.lower()
            operation = f'llama_index.op.{event_name_str}'

            if event_type == CBEventType.EMBEDDING:
                span = self._start_trace(event_id, operation)
            elif event_type == CBEventType.LLM:
                span = self._start_trace(event_id, operation)
                if span:
                    if payload and 'template' in payload:
                        span.set_payload('template', payload['template'])
                    if payload and 'context_str' in payload:
                        span.set_payload('context', payload['context_str'])
            elif event_type == CBEventType.QUERY:
                span = self._start_trace(event_id, operation)
            elif event_type == CBEventType.RETRIEVE:
                span = self._start_trace(event_id, operation)
            elif event_type == CBEventType.SYNTHESIZE:
                span = self._start_trace(event_id, operation)
        except Exception:
            logger.error('Error in LlamaIndex callback handler', exc_info=True)

    def on_event_end(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any
    ) -> None:
        try:
            span = self._current_span(event_id)
            if span:
                if event_type == CBEventType.EMBEDDING:
                    pass
                elif event_type == CBEventType.LLM:
                    if payload:
                        if 'formatted_prompt' in payload:
                            span.set_payload('formatted_prompt', payload['formatted_prompt'])
                        if 'response' in payload:
                            span.set_payload('response', payload['response'])
                        if 'completion' in payload:
                            span.set_payload('completion', payload['completion'])
                elif event_type == CBEventType.QUERY:
                    pass
                elif event_type == CBEventType.RETRIEVE:
                    if payload and 'nodes' in payload:
                        span.set_payload('nodes',payload['nodes'])
                elif event_type == CBEventType.SYNTHESIZE:
                    if payload and 'response' in payload:
                        span.set_payload('response', str(payload['response']))
                self._stop_trace(event_id)
        except Exception:
            logger.error('Error in LlamaIndex callback handler', exc_info=True)

    def start_trace(self, trace_id: Optional[str] = None) -> None:
        try:
            if trace_id is not None:
                trace_map = _llama_trace_map.get()
                if trace_id not in trace_map:
                    operation = f'llama_index.{trace_id}' if trace_id else 'llama_index.root'
                    span = graphsignal.trace(operation)
                    trace_map[trace_id] = span
                    _llama_trace_map.set(trace_map)
        except Exception:
            logger.error('Error in LlamaIndex callback handler', exc_info=True)

    def end_trace(
        self,
        trace_id: Optional[str] = None,
        trace_map: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        try:
            if trace_id is not None:
                trace_map = _llama_trace_map.get()
                span = trace_map.pop(trace_id)
                if span is not None:
                    span.stop()
                    _llama_trace_map.set(trace_map)
        except Exception:
            logger.error('Error in LlamaIndex callback handler', exc_info=True)
