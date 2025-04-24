from typing import Union, Any, Optional, Dict
import logging
import sys
import time
import traceback
import random
import contextvars

import graphsignal
from graphsignal import client
from graphsignal.utils import uuid_sha1, sanitize_str

logger = logging.getLogger('graphsignal')

RANDOM_CACHE = [random.random() for _ in range(10000)]
idx = 0

def fast_random():
    global idx
    idx = (idx + 1) % len(RANDOM_CACHE)
    return RANDOM_CACHE[idx]

def _tracer():
    return graphsignal._tracer

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
        'scope',
        'name',
        'value',
        'unit']

    def __init__(self, scope, name, value, unit=None):
        self.scope = scope
        self.name = name
        self.value = value
        self.unit = unit

trace_context_var = contextvars.ContextVar('gsig_trace_ctx', default=[])

class SpanContext:
    __slots__ = [
        'trace_id',
        'span_id',
        'sampled',
        'profiled'
    ]

    def __init__(self, trace_id=None, span_id=None, sampled=None, profiled=None):
        self.trace_id = trace_id
        self.span_id = span_id
        self.sampled = sampled
        self.profiled = profiled

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
        if len(parts) != 4:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'SpanContext.loads: invalid context value: {value}')
            return None
        ctx = SpanContext()
        ctx.trace_id = parts[0]
        ctx.span_id = parts[1]
        ctx.sampled = parts[2] == '1'
        ctx.profiled = parts[3] == '1'
        return ctx

    @staticmethod
    def dumps(ctx):
        if ctx is None or ctx.trace_id is None or ctx.span_id is None or ctx.sampled is None or ctx.profiled is None:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('SpanContext.dumps: invalid context')
            return None
        return '{0}-{1}-{2}-{3}'.format(
            ctx.trace_id, 
            ctx.span_id, 
            '1' if ctx.sampled else '0', 
            '1' if ctx.profiled else '0')

class Span:
    MAX_SPAN_TAGS = 25
    MAX_PARAMS = 100
    MAX_USAGES_COUNTERS = 25
    MAX_PAYLOADS = 10
    MAX_PAYLOAD_BYTES = 256 * 1024
    MAX_PROFILES = 10
    MAX_PROFILE_SIZE = 256 * 1024

    __slots__ = [
        '_operation',
        '_tags',
        '_with_profile',
        '_context_tags',
        '_span_id',
        '_trace_id',
        '_parent_span_id',
        '_linked_span_ids',
        '_is_root',
        '_sampled',
        '_profiled',
        '_recorder_context',
        '_model',
        '_is_started',
        '_is_stopped',
        '_start_us',
        '_start_counter',
        '_stop_counter',
        '_first_token_counter',
        '_output_tokens',
        '_exc_infos',
        '_params',
        '_counters',
        '_profiles',
        '_metrics'
    ]

    def __init__(self, operation, tags=None, with_profile=False, parent_context=None):
        self._is_started = False

        if not operation:
            logger.error('Span: operation is required')
            return
        if tags is not None:
            if not isinstance(tags, dict):
                logger.error('Span: tags must be dict')
                return
            if len(tags) > Span.MAX_SPAN_TAGS:
                logger.error('Span: too many tags (>{0})'.format(Span.MAX_SPAN_TAGS))
                return

        self._operation = sanitize_str(operation)
        self._tags = dict(operation=self._operation)
        if tags is not None:
            self._tags.update(tags)
        self._with_profile = with_profile
        self._context_tags = None
        self._is_stopped = False
        self._span_id = None
        self._trace_id = None
        self._parent_span_id = None
        self._sampled = None
        self._profiled = None
        if parent_context:
            self._trace_id = parent_context.trace_id
            self._parent_span_id = parent_context.span_id
            self._sampled = parent_context.sampled
            self._profiled = parent_context.profiled
        self._linked_span_ids = None
        self._is_root = False
        self._start_us = None
        self._start_counter = None
        self._stop_counter = None
        self._first_token_counter = None
        self._output_tokens = None
        self._recorder_context = False
        self._model = None
        self._exc_infos = None
        self._counters = None
        self._params = None
        self._profiles = None
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

        if _tracer().debug_mode:
            logger.debug(f'Starting span {self._operation}')

        self._span_id = uuid_sha1(size=12)
        if self._trace_id is None:
            self._trace_id = uuid_sha1(size=12)
            self._is_root = True
        if self._sampled is None:
            self._sampled = fast_random() <= graphsignal._tracer.sampling_rate
        if self._sampled and self._profiled is None:
            self._profiled = fast_random() <= graphsignal._tracer.profiling_rate

        self._context_tags = _tracer().context_tags.get().copy()

        self._model = client.Span(
            span_id=self._span_id,
            start_us=0,
            end_us=0,
            tags=[],
            exceptions=[],
            params=[],
            counters=[],
            profiles=[]
        )

        self._recorder_context = {}

        # emit start event
        try:
            _tracer().emit_span_start(self, self._recorder_context)
        except Exception as exc:
            logger.error('Error in span start event handlers', exc_info=True)

        self._start_us = int(time.time() * 1e6)
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

        if _tracer().debug_mode:
            logger.debug(f'Stopping span {self._operation}')

        if self._stop_counter is None:
            self._measure()
        latency_ns = self._stop_counter - self._start_counter
        end_us = int(self._start_us + latency_ns / 1e3)
        now = int(end_us / 1e6)

        self.set_counter('latency_ns', latency_ns)        

        # emit stop event
        try:
            _tracer().emit_span_stop(self, self._recorder_context)
        except Exception as exc:
            logger.error('Error in span stop event handlers', exc_info=True)

        # emit read event
        try:
            _tracer().emit_span_read(self, self._recorder_context)
        except Exception as exc:
            logger.error('Error in span read event handlers', exc_info=True)

        span_tags = self._merged_span_tags()

        # update RED metrics
        first_token_ns = self.get_counter('first_token_ns')
        output_tokens = self.get_counter('output_tokens')

        _tracer().metric_store().inc_counter(
            scope='performance', name='call_count', tags=span_tags, value=1, update_ts=now)

        if self._exc_infos and len(self._exc_infos) > 0:
            for exc_info in self._exc_infos:
                if exc_info[0] is not None:
                    _tracer().metric_store().inc_counter(
                        scope='performance', name='exception_count', tags=span_tags, value=1, update_ts=now)
                    self.set_tag('exception', exc_info[0].__name__)

        if latency_ns is not None:
            _tracer().metric_store().update_histogram(
                scope='performance', name='latency', tags=span_tags, value=latency_ns, update_ts=now, is_time=True)

        if first_token_ns is not None:
            _tracer().metric_store().update_histogram(
                scope='performance', name='first_token', tags=span_tags, value=first_token_ns, update_ts=now, is_time=True)

        if latency_ns is not None and latency_ns > 0 and output_tokens is not None and output_tokens > 0:
            _tracer().metric_store().update_rate(
                scope='performance', 
                name='output_tps', 
                tags=span_tags, 
                count=output_tokens, 
                interval=latency_ns/1e9, 
                update_ts=now)

        # update metrics
        if self._metrics is not None:
            for metric in self._metrics.values():
                _tracer().metric_store().inc_counter(
                    scope=metric.scope, name=metric.name, tags=span_tags, value=metric.value, update_ts=now)

        # update recorder metrics
        if _tracer().check_metric_read_interval(now):
            _tracer().set_metric_read(now)
            try:
                _tracer().emit_metric_update()
            except Exception as exc:
                logger.error('Error in span read event handlers', exc_info=True)

        if self._sampled:
            # fill and upload span
            # copy data to span model
            self._model.start_us = self._start_us
            self._model.end_us = end_us
            if self._trace_id:
                self._model.trace_id = self._trace_id
            if self._parent_span_id:
                self._model.parent_span_id = self._parent_span_id
            if self._linked_span_ids:
                self._model.linked_span_ids = self._linked_span_ids

            # copy tags
            for key, value in span_tags.items():
                self._model.tags.append(client.Tag(
                    key=sanitize_str(key, max_len=50),
                    value=sanitize_str(value, max_len=250)
                ))

            # copy exception
            if self._exc_infos:
                for exc_info in self._exc_infos:
                    exc_type = None
                    message = None
                    stack_trace = None
                    if exc_info[0] and hasattr(exc_info[0], '__name__'):
                        exc_type = str(exc_info[0].__name__)
                    if exc_info[1]:
                        message = str(exc_info[1])
                    if exc_info[2]:
                        frames = traceback.format_tb(exc_info[2])
                        if len(frames) > 0:
                            stack_trace = ''.join(frames)

                    if exc_type and message:
                        exception_model = client.Exception(
                            exc_type=exc_type,
                            message=message,
                        )
                        if stack_trace:
                            exception_model.stack_trace = stack_trace
                        self._model.exceptions.append(exception_model)

            # copy params
            if self._params is not None:
                for key, value in self._merged_params().items():
                    self._model.params.append(client.Param(
                        name=sanitize_str(key, max_len=50),
                        value=sanitize_str(value, max_len=250)
                    ))

            # copy counters
            if self._counters is not None:
                for counter in self._counters.values():
                    self._model.counters.append(client.Counter(
                        name=counter.name,
                        value=counter.value
                    ))

            # copy profiles
            if self._profiles is not None:
                for profile in self._profiles.values():
                    if len(profile.content) <= Span.MAX_PROFILE_SIZE:
                        self._model.profiles.append(client.Profile(
                            name=profile.name,
                            format=profile.format,
                            content=profile.content
                        ))

            # queue span model for upload
            _tracer().uploader().upload_span(self._model)

        # trigger upload
        if self._is_root:
            _tracer().tick(now)

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
    
    def get_span_context(self):
        return SpanContext(
            trace_id=self._trace_id,
            span_id=self._span_id,
            sampled=self._sampled,
            profiled=self._profiled)

    def add_linked_span(self, span_id: str) -> None:
        if not span_id:
            logger.error('add_linked_span: span_id must be provided')
            return

        if self._linked_span_ids is None:
            self._linked_span_ids = []

        self._linked_span_ids.append(span_id)

    def sampled(self) -> bool:
        return self._sampled

    def profiled(self) -> bool:
        return self._profiled

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

    def set_param(self, name: str, value: str) -> None:
        if self._params is None:
            self._params = {}

        if not name:
            logger.error('set_param: name must be provided')
            return

        if not value:
            logger.error('set_param: value must be provided')
            return

        if len(self._params) > Span.MAX_PARAMS:
            logger.error('set_param: too many params (>{0})'.format(Span.MAX_PARAMS))
            return

        self._params[name] = value

    def get_param(self, name: str) -> Optional[str]:
        if self._params is None:
            return None
        return self._params.get(name)

    def set_counter(self, name: str, value: int) -> None:
        if self._counters is None:
            self._counters = {}

        if name and not isinstance(name, str):
            logger.error('set_counter: name must be string')
            return

        if value and not isinstance(value, (int, float)):
            logger.error('set_counter: value must be number')
            return

        if len(self._counters) > Span.MAX_USAGES_COUNTERS:
            logger.error('set_counter: too many counters (>{0})'.format(Span.MAX_USAGES_COUNTERS))
            return

        self._counters[name] = Counter(name=name, value=value)

    def inc_counter(self, name: str, value: Union[int, float]) -> None:
        if self._counters is None or name not in self._counters:
            self.set_counter(name, value)
        else:
            self._counters[name].value += value

    def set_perf_counter(self, name: str) -> None:
        if not self._is_stopped:
            self.set_counter(name, time.perf_counter_ns() - self._start_counter)

    def get_counter(self, name: str) -> Optional[Union[int, float]]:
        if self._counters is None:
            return None
        counter = self._counters.get(name)
        if counter:
            return counter.value
        return None

    def set_profile(
            self, 
            name: str, 
            format: str,
            content: str) -> None:
        if self._profiles is None:
            self._profiles = {}

        if not name or not isinstance(name, str):
            logger.error('set_profile: name must be string')
            return

        if len(self._profiles) > Span.MAX_PROFILES:
            logger.error('set_profile: too many profiles (>{0})'.format(Span.MAX_PROFILES))
            return

        self._profiles[name] = Profile(
            name=name,
            format=format,
            content=content)

    def score(
            self,
            name: str, 
            score: Union[int, float], 
            unit: Optional[str] = None,
            severity: Optional[int] = None,
            comment: Optional[str] = None) -> None:
        now = int(time.time())

        if not name:
            logger.error('Span.score: name is required')
            return

        if not name:
            logger.error('Span.score: score is required')
            return

        score_obj = client.Score(
            score_id=uuid_sha1(size=12),
            tags=[],
            span_id=self._span_id,
            name=name,
            score=score,
            create_ts=now
        )

        for tag_key, tag_value in self._merged_span_tags().items():
            score_obj.tags.append(client.Tag(
                key=sanitize_str(tag_key, max_len=50),
                value=sanitize_str(tag_value, max_len=250)
            ))

        if unit is not None:
            score_obj.unit = unit

        if severity and severity >= 1 and severity <= 5:
            score_obj.severity = severity

        if comment:
            score_obj.comment = comment
        
        _tracer().uploader().upload_score(score_obj)
        _tracer().tick(now)

    def trace(
            self, 
            operation: str,
            tags: Optional[Dict[str, str]] = None) -> 'Span':
        return Span(
            operation=operation, 
            tags=tags,
            parent_context=self.get_span_context())
 
    def inc_counter_metric(
            self,
            scope: str,
            name: str,
            value: Union[int, float],
            unit=None) -> None:
        if not scope:
            logger.error('inc_counter_metric: scope is required')
            return
        if not name:
            logger.error('inc_counter_metric: name is required')
            return
        if value is None and not isinstance(value, (str, float)):
            logger.error('inc_counter_metric: value is required')
            return
        
        if self._metrics is None:
            self._metrics = {}

        key = '{0}-{1}'.format(scope, name)
        if key in self._metrics:
            metric = self._metrics[key]
            metric.value += value
        else:
            metric = CounterMetric(
                scope=scope,
                name=name,
                value=value,
                unit=unit)
            self._metrics[key] = metric

    def _merged_span_tags(self, extra_tags=None):
        span_tags = {}
        if _tracer().tags is not None:
            span_tags.update(_tracer().tags)
        if self._context_tags:
            span_tags.update(self._context_tags)
        if self._tags is not None:
            span_tags.update(self._tags)
        if extra_tags is not None:
            span_tags.update(extra_tags)
        return span_tags

    def _merged_params(self):
        params = {}
        if _tracer().params is not None:
            params.update(_tracer().params)
        if self._params is not None:
            params.update(self._params)
        return params

    def repr(self):
        return 'Span({0})'.format(self._operation)
