from typing import Union, Any, Optional, Dict
import logging
import sys
import time
import contextvars
import traceback

import graphsignal
from graphsignal.proto import signals_pb2
from graphsignal.utils import uuid_sha1, sanitize_str

logger = logging.getLogger('graphsignal')

def _ticker():
    return graphsignal._ticker

class Counter:
    __slots__ = [
        'name',
        'value'
    ]

    def __init__(self, name, value):
        self.name = name
        self.value = value

class Profile:
    __slots__ = [
        'name',
        'format',
        'content']

    def __init__(self, name, format, content):
        self.name = name
        self.format = format
        self.content = content

class CounterMetric:
    __slots__ = [
        'name',
        'value',
        'unit']

    def __init__(self, name, value, unit=None):
        self.name = name
        self.value = value
        self.unit = unit

trace_context_var = contextvars.ContextVar('gsig_trace_ctx', default=[])

class SpanContext:
    __slots__ = [
        'trace_id',
        'span_id',
        'sampled'
    ]

    def __init__(self, trace_id=None, span_id=None, sampled=False):
        self.trace_id = trace_id
        self.span_id = span_id
        self.sampled = sampled

    @staticmethod
    def push_contextvars(ctx):
        trace_context_var.set(trace_context_var.get() + [SpanContext.dumps(ctx)])

    @staticmethod
    def pop_contextvars():
        trace_context = trace_context_var.get()
        if len(trace_context) > 0:
            ctx = SpanContext.loads(trace_context[-1])
            trace_context_var.set(trace_context[:-1])
            return ctx

    @staticmethod
    def loads(value):
        if value is None:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'SpanContext.loads: invalid context value: {value}')
            return None
        parts = value.split('-')
        if len(parts) < 2:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'SpanContext.loads: invalid context value: {value}')
            return None
        ctx = SpanContext()
        ctx.trace_id = parts[0]
        ctx.span_id = parts[1]
        ctx.sampled = parts[2] == '1'
        return ctx

    @staticmethod
    def dumps(ctx):
        if ctx is None or ctx.trace_id is None or ctx.span_id is None or ctx.sampled is None:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('SpanContext.dumps: invalid context')
            return None
        return '{0}-{1}-{2}'.format(
            ctx.trace_id, 
            ctx.span_id,
            '1' if ctx.sampled else '0')

class Span:
    MAX_SPAN_TAGS = 25
    MAX_ATTRIBUTES = 100
    MAX_COUNTERS = 25
    MAX_PROFILES = 10
    MAX_PROFILE_SIZE = 100 * 1024 * 1024

    __slots__ = [
        '_name',
        '_tags',
        '_context_tags',
        '_span_id',
        '_trace_id',
        '_parent_span_id',
        '_linked_span_ids',
        '_is_root',
        '_sampled',
        '_recorder_context',
        '_proto',
        '_is_started',
        '_is_stopped',
        '_start_ts',
        '_start_counter',
        '_stop_counter',
        '_first_token_counter',
        '_output_tokens',
        '_exc_infos',
        '_attributes',
        '_counters',
        '_metrics'
    ]

    def __init__(self, name, tags=None, parent_context=None):
        self._is_started = False

        if not name:
            logger.error('Span: name is required')
            return
        if tags is not None:
            if not isinstance(tags, dict):
                logger.error('Span: tags must be dict')
                return
            if len(tags) > Span.MAX_SPAN_TAGS:
                logger.error('Span: too many tags (>{0})'.format(Span.MAX_SPAN_TAGS))
                return

        self._name = sanitize_str(name)
        self._tags = None
        if tags is not None:
            self._tags = {}
            self._tags.update(tags)
        self._context_tags = None
        self._is_stopped = False
        self._span_id = None
        self._trace_id = None
        self._parent_span_id = None
        self._sampled = False
        if parent_context:
            self._trace_id = parent_context.trace_id
            self._parent_span_id = parent_context.span_id
            self._sampled = parent_context.sampled
        self._linked_span_ids = None
        self._is_root = False
        self._start_ts = None
        self._start_counter = None
        self._stop_counter = None
        self._first_token_counter = None
        self._output_tokens = None
        self._recorder_context = False
        self._proto = None
        self._exc_infos = None
        self._counters = None
        self._attributes = None
        self._metrics = None

        try:
            self._start()
        except Exception:
            logger.error('Error starting span', exc_info=True)
            self._is_stopped = True

    def __enter__(self):
        return self

    async def __aenter__(self):
        return self

    def __exit__(self, *exc_info):
        if exc_info and exc_info[1] and isinstance(exc_info[1], Exception):
            if not self._exc_infos:
                self._exc_infos = []
            self._exc_infos.append(exc_info)
        self.stop()
        return False

    async def __aexit__(self, *exc_info):
        if exc_info and exc_info[1] and isinstance(exc_info[1], Exception):
            if not self._exc_infos:
                self._exc_infos = []
            self._exc_infos.append(exc_info)
        self.stop()
        return False

    def _start(self):
        if self._is_started:
            return
        if self._is_stopped:
            return

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'Starting span {self._name}')

        self._span_id = uuid_sha1(size=12)
        if self._trace_id is None:
            self._trace_id = uuid_sha1(size=12)
            self._is_root = True
        
        self._context_tags = _ticker().context_tags.get().copy()

        self._proto = signals_pb2.Span()
        self._proto.span_id = self._span_id
        self._proto.trace_id = self._trace_id
        self._proto.start_ts = 0
        self._proto.end_ts = 0
        self._proto.name = self._name

        self._recorder_context = {}

        if not self.is_sampled():
            if _ticker().should_trace((self._name, 'random')):
                self.set_sampled(True)
                self.set_tag('sampling.reason', 'span.random')
        else:
            if self._parent_span_id:
                self.set_tag('sampling.reason', 'span.parent')
        
        self._start_ts = time.time_ns()
        self._start_counter = time.perf_counter_ns()
        self._is_started = True

    def _measure(self) -> None:
        self._stop_counter = time.perf_counter_ns()

    def _stop(self) -> None:
        if not self._is_started:
            return
        if self._is_stopped:
            return
        self._is_stopped = True

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'Stopping span {self._name}')

        if self._stop_counter is None:
            self._measure()
        duration_ns = self._stop_counter - self._start_counter
        end_ns = int(self._start_ts + duration_ns)

        self.set_counter('span.duration', duration_ns)

        # update RED metrics
        metric_tags = {'span.name': self._name}

        _ticker().inc_counter(
            name='span.call.count', tags=metric_tags, value=1, measurement_ts=end_ns, aggregate=True)

        if self._exc_infos and len(self._exc_infos) > 0:
            for exc_info in self._exc_infos:
                if exc_info[0] is not None:
                    _ticker().inc_counter(
                        name='span.error.count', tags=metric_tags, value=1, measurement_ts=end_ns, aggregate=True)
                    self.set_tag('exception.name', exc_info[0].__name__)

        if duration_ns is not None:
            _ticker().update_histogram(
                name='span.duration', tags=metric_tags, value=duration_ns, measurement_ts=end_ns, aggregate=True)

        # update metrics
        if self._metrics is not None:
            for metric in self._metrics.values():
                _ticker().inc_counter(
                    name=metric.name, tags=metric_tags, value=metric.value, measurement_ts=end_ns, aggregate=True)

        # add exception events
        if self._exc_infos:
            for exc_info in self._exc_infos:
                if _ticker().should_trace((self._name, 'error')):
                    self.set_sampled(True)
                    self.set_tag('sampling.reason', 'span.error')

                if self._sampled:
                    self.set_tag('span.status', 'error')

                    event = self._proto.events.add()
                    event.name = 'exception'
                    event.event_ts = end_ns
                    
                    if exc_info[0] and hasattr(exc_info[0], '__name__'):
                        attr = event.attributes.add()
                        attr.name = 'exception.type'
                        attr.value = sanitize_str(exc_info[0].__name__, max_len=2500)
                    
                    if exc_info[1]:
                        message = str(exc_info[1])
                        attr = event.attributes.add()
                        attr.name = 'exception.message'
                        attr.value = sanitize_str(message, max_len=2500)
                    
                    if exc_info[2]:
                        frames = traceback.format_tb(exc_info[2])
                        if len(frames) > 0:
                            stack_trace = ''.join(frames)
                            attr = event.attributes.add()
                            attr.name = 'exception.stacktrace'
                            attr.value = sanitize_str(stack_trace, max_len=2500)

        if self._sampled:
            # fill and upload span
            # copy data to span proto
            self._proto.start_ts = self._start_ts
            self._proto.end_ts = end_ns
            self._proto.name = self._name
            if self._parent_span_id:
                self._proto.parent_span_id = self._parent_span_id
            if self._linked_span_ids:
                self._proto.linked_span_ids.extend(self._linked_span_ids)

            # copy tags
            span_tags = self._merged_span_tags()
            for key, value in span_tags.items():
                tag = self._proto.tags.add()
                tag.key = sanitize_str(key, max_len=50)
                tag.value = sanitize_str(value, max_len=250)

            # copy attributes
            if self._attributes is not None:
                for key, value in self._attributes.items():
                    attr = self._proto.attributes.add()
                    attr.name = sanitize_str(key, max_len=50)
                    attr.value = sanitize_str(value, max_len=2500)

            # copy counters
            if self._counters is not None:
                for counter in self._counters.values():
                    cnt = self._proto.counters.add()
                    cnt.name = counter.name
                    cnt.value = counter.value

            # queue span proto for upload
            _ticker().signal_uploader().upload_span(self._proto)

    def measure(self) -> None:
        if not self._is_stopped:
            self._measure()

    def stop(self) -> None:
        try:
            self._stop()
        except Exception:
            logger.error('Error stopping span', exc_info=True)
        finally:
            self._is_stopped = True
    
    @property
    def span_id(self) -> str:
        return self._span_id

    @property
    def trace_id(self) -> str:
        return self._trace_id

    @property
    def name(self) -> str:
        return self._name

    def get_span_context(self):
        return SpanContext(
            trace_id=self._trace_id,
            span_id=self._span_id,
            sampled=self._sampled)

    def add_linked_span(self, span_id: str) -> None:
        if not span_id:
            logger.error('add_linked_span: span_id must be provided')
            return

        if self._linked_span_ids is None:
            self._linked_span_ids = []

        self._linked_span_ids.append(span_id)

    def set_sampled(self, sampled: bool) -> None:
        self._sampled = sampled

    def is_sampled(self) -> bool:
        return self._sampled

    def set_tag(self, key: str, value: str) -> None:
        if not key:
            logger.error('set_tag: key must be provided')
            return

        if self._tags is None:
            self._tags = {}

        if value is None:
            self._tags.pop(key, None)
            return

        if len(self._tags) > Span.MAX_SPAN_TAGS:
            logger.error('set_tag: too many tags (>{0})'.format(Span.MAX_SPAN_TAGS))
            return

        self._tags[key] = value

    def get_tag(self, key) -> Optional[str]:
        if self._tags is None:
            return None
        return self._tags.get(key)
    
    def get_tags(self) -> Dict[str, str]:
        if self._tags is None:
            return {}
        return self._tags.copy()

    def add_exception(self, exc: Optional[Exception] = None, exc_info: Optional[bool] = None) -> None:
        if exc is not None and not isinstance(exc, Exception):
            logger.error('add_exception: exc must be instance of Exception')
            return

        if exc_info is not None and not isinstance(exc_info, bool):
            logger.error('add_exception: exc_info must be bool')
            return

        if self._exc_infos is None:
            self._exc_infos = []

        if exc:
            self._exc_infos.append((exc.__class__, str(exc), exc.__traceback__))
        elif exc_info == True:
            self._exc_infos.append(sys.exc_info())

    def set_attribute(self, name: str, value: Any) -> None:
        if self._attributes is None:
            self._attributes = {}

        if not name:
            logger.error('set_attribute: name must be provided')
            return

        if not value:
            logger.error('set_attribute: value must be provided')
            return

        if len(self._attributes) > Span.MAX_ATTRIBUTES:
            logger.error('set_attribute: too many attributes (>{0})'.format(Span.MAX_ATTRIBUTES))
            return

        self._attributes[name] = value

    def get_attribute(self, name: str) -> Optional[str]:
        if self._attributes is None:
            return None
        return self._attributes.get(name)

    def set_counter(self, name: str, value: int) -> None:
        if self._counters is None:
            self._counters = {}

        if name and not isinstance(name, str):
            logger.error('set_counter: name must be string')
            return

        if value and not isinstance(value, (int, float)):
            logger.error('set_counter: value must be number')
            return

        if len(self._counters) > Span.MAX_COUNTERS:
            logger.error('set_counter: too many counters (>{0})'.format(Span.MAX_COUNTERS))
            return

        self._counters[name] = Counter(name=name, value=value)

    def inc_counter(self, name: str, value: Union[int, float]) -> None:
        if self._counters is None or name not in self._counters:
            self.set_counter(name, value)
        else:
            self._counters[name].value += value

    def get_counter(self, name: str) -> Optional[Union[int, float]]:
        if self._counters is None:
            return None
        counter = self._counters.get(name)
        if counter:
            return counter.value
        return None

    def measure_event_as_counter(self, name: str) -> None:
        elapsed_ns = time.perf_counter_ns() - self._start_counter
        self.set_counter(name, elapsed_ns)

    def trace(
            self, 
            span_name: str,
            tags: Optional[Dict[str, str]] = None) -> 'Span':
        return Span(
            name=span_name, 
            tags=tags,
            parent_context=self.get_span_context())
 
    def inc_counter_metric(
            self,
            name: str,
            value: Union[int, float],
            unit=None) -> None:
        if not name:
            logger.error('inc_counter_metric: name is required')
            return
        if not value or not isinstance(value, (int, float)):
            logger.error('inc_counter_metric: value is required')
            return
        
        if self._metrics is None:
            self._metrics = {}

        if name in self._metrics:
            metric = self._metrics[name]
            metric.value += value
        else:
            metric = CounterMetric(
                name=name,
                value=value,
                unit=unit)
            self._metrics[name] = metric

    def _merged_span_tags(self, extra_tags=None):
        span_tags = {}
        if _ticker().tags is not None:
            span_tags.update(_ticker().tags)
        if self._context_tags:
            span_tags.update(self._context_tags)
        if self._tags is not None:
            span_tags.update(self._tags)
        if extra_tags is not None:
            span_tags.update(extra_tags)
        return span_tags

    def repr(self):
        return 'Span({0})'.format(self._name)
