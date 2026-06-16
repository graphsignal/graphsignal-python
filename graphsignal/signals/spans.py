from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import logging
import time
from collections import OrderedDict

import graphsignal
import graphsignal.sdk
from graphsignal.proto import signals_pb2

logger = logging.getLogger('graphsignal')


def sanitize_str(val, max_len=250):
    if not isinstance(val, str):
        return str(val)[:max_len]
    return val[:max_len]


@dataclass
class _TraceState:
    """Per-trace export sampling (traces_per_sec) and span export limits."""
    export_sampled: Optional[bool] = None
    total: int = 0
    per_op: Dict[str, int] = field(default_factory=dict)  # keyed by span op_name


class TokenBucketSampler:
    """Token-bucket sampler: ``rate_per_sec`` refill, burst up to ``capacity`` tokens."""

    def __init__(self, rate_per_sec: float, *, capacity: Optional[float] = None):
        if not (0.001 <= rate_per_sec <= 1000):
            raise ValueError("rate_per_sec must be in [0.001, 1000].")
        self.rate_per_sec = float(rate_per_sec)
        self.capacity = max(1.0, float(capacity if capacity is not None else rate_per_sec))
        self._tokens = self.capacity
        self._last_refill_ms: Optional[int] = None

    @staticmethod
    def _now_ms() -> int:
        return time.time_ns() // 1_000_000

    def _refill(self, now_ms: int) -> None:
        if self._last_refill_ms is None:
            self._last_refill_ms = now_ms
            return
        elapsed_sec = (now_ms - self._last_refill_ms) / 1000.0
        if elapsed_sec <= 0:
            return
        self._tokens = min(
            self.capacity, self._tokens + elapsed_sec * self.rate_per_sec)
        self._last_refill_ms = now_ms

    def should_sample(self) -> bool:
        now_ms = self._now_ms()
        self._refill(now_ms)
        if self._tokens >= 1.0:
            self._tokens -= 1.0
            return True
        return False


class SpanStore:
    """Holds span protos pending upload, a global trace-rate sampler, and
    per-trace export sampling state.
    """

    MAX_TRACE_STATES = 10000
    MAX_SPANS_PER_TRACE = 256
    MAX_SPANS_PER_OP_PER_TRACE = 32
    MAX_INITIAL_TRACE_EXPORTS = 10

    def __init__(self, config_loader=None):
        self._config_loader = config_loader
        self._sampler: Optional[TokenBucketSampler] = None
        self._trace_states: "OrderedDict[str, _TraceState]" = OrderedDict()
        self._spans: List[signals_pb2.Span] = []
        self._warned_no_rate = False
        self._initial_trace_exports_remaining = self.MAX_INITIAL_TRACE_EXPORTS

    def reset_samplers(self) -> None:
        self._sampler = None
        self._warned_no_rate = False

    def _ensure_sampler(self) -> Optional[TokenBucketSampler]:
        if self._sampler is not None:
            return self._sampler

        if self._config_loader is None:
            return None
        traces_per_sec = self._config_loader.get_float_option('traces_per_sec')
        if not traces_per_sec:
            if not self._warned_no_rate:
                self._warned_no_rate = True
                logger.warning(
                    'Spans received but traces_per_sec is not configured '
                    '(server-side sampling rate); dropping all spans until it is set')
            return None

        self._sampler = TokenBucketSampler(traces_per_sec)
        return self._sampler

    def _trace_export_draw_allows(self) -> bool:
        if self._initial_trace_exports_remaining > 0:
            self._initial_trace_exports_remaining -= 1
            return True
        sampler = self._ensure_sampler()
        return bool(sampler and sampler.should_sample())

    def _purge_queued_spans_for_trace(self, trace_id: str) -> None:
        sanitized_trace_id = sanitize_str(trace_id, max_len=64)
        if self._spans:
            self._spans = [
                s for s in self._spans if s.trace_id != sanitized_trace_id]
        state = self._trace_states.get(trace_id)
        if state is not None:
            state.total = 0
            state.per_op.clear()

    def _trace_state(self, trace_id: str) -> _TraceState:
        state = self._trace_states.get(trace_id)
        if state is None:
            state = _TraceState()
            self._trace_states[trace_id] = state
        self._trace_states.move_to_end(trace_id)
        while len(self._trace_states) > self.MAX_TRACE_STATES:
            evicted_id, _ = self._trace_states.popitem(last=False)
            self._purge_queued_spans_for_trace(evicted_id)
        return state

    def _export_allowed(self, trace_id: str, op_name: str) -> bool:
        state = self._trace_state(trace_id)
        if state.export_sampled is False:
            return False
        if state.export_sampled is None:
            state.export_sampled = self._trace_export_draw_allows()
            if not state.export_sampled:
                return False

        if state.total >= self.MAX_SPANS_PER_TRACE:
            return False
        if state.per_op.get(op_name, 0) >= self.MAX_SPANS_PER_OP_PER_TRACE:
            return False

        state.per_op[op_name] = state.per_op.get(op_name, 0) + 1
        state.total += 1
        return True

    def record_span(
            self,
            *,
            trace_id: str,
            span_id: str,
            name: str,
            start_ts: int,
            end_ts: int,
            parent_span_id: Optional[str] = None,
            attributes: Optional[Dict[str, Any]] = None,
            events: Optional[List[Dict[str, Any]]] = None,
            tags: Optional[Dict[str, str]] = None,
    ) -> Optional[signals_pb2.Span]:
        """Convert a span into a `signals_pb2.Span` proto and enqueue it.

        Returns the proto on success, or None if dropped (invalid input or not
        sampled). The caller can use the return value to feed downstream
        consumers (e.g. the span profiler).
        """
        if not name or not trace_id or not span_id:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('SpanStore.record_span: invalid span (name=%r trace_id=%r span_id=%r)',
                             name, trace_id, span_id)
            return None
        if not (start_ts > 0 and end_ts > 0):
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('SpanStore.record_span: invalid timestamps (start=%s end=%s)',
                             start_ts, end_ts)
            return None

        op_name = sanitize_str(name)
        if not self._export_allowed(trace_id, op_name):
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    'SpanStore.record_span: not exported (trace_id=%r name=%r)',
                    trace_id, name)
            return None

        span = signals_pb2.Span()
        span.span_id = sanitize_str(span_id, max_len=64)
        span.trace_id = sanitize_str(trace_id, max_len=64)
        if parent_span_id:
            span.parent_span_id = sanitize_str(parent_span_id, max_len=64)
        span.start_ts = int(start_ts)
        span.end_ts = int(end_ts)
        span.name = op_name

        duration_ns = int(end_ts) - int(start_ts)
        if duration_ns > 0:
            counter = span.counters.add()
            counter.name = 'span.duration'
            counter.value = float(duration_ns)

        for tag_key, tag_value in graphsignal.sdk.sdk().tags().items():
            tag = span.tags.add()
            tag.key = sanitize_str(tag_key, max_len=50)
            tag.value = sanitize_str(tag_value, max_len=250)

        if tags:
            for tag_key, tag_value in tags.items():
                tag = span.tags.add()
                tag.key = sanitize_str(tag_key, max_len=50)
                tag.value = sanitize_str(tag_value, max_len=250)

        if attributes:
            for attr_name, attr_value in attributes.items():
                if attr_value is None:
                    continue
                attr = span.attributes.add()
                attr.name = sanitize_str(attr_name, max_len=50)
                attr.value = sanitize_str(attr_value, max_len=2500)

        if events:
            for event in events:
                event_proto = span.events.add()
                event_proto.name = sanitize_str(event.get('name', ''), max_len=50)
                event_proto.event_ts = int(event.get('event_ts', 0) or 0)
                for attr_name, attr_value in (event.get('attributes') or {}).items():
                    if attr_value is None:
                        continue
                    attr = event_proto.attributes.add()
                    attr.name = sanitize_str(attr_name, max_len=50)
                    attr.value = sanitize_str(attr_value, max_len=2500)

        self._spans.append(span)
        return span

    def has_unexported(self) -> bool:
        return len(self._spans) > 0

    def export(self) -> List[signals_pb2.Span]:
        spans = self._spans
        self._spans = []
        for state in self._trace_states.values():
            state.total = 0
            state.per_op.clear()
        return spans

    def clear(self) -> None:
        self._spans = []
        self._trace_states.clear()
        self._sampler = None
        self._initial_trace_exports_remaining = self.MAX_INITIAL_TRACE_EXPORTS
