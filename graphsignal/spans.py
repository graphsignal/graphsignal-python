from typing import Union, Any, Optional, Dict
import logging
import sys
import time
import traceback
import contextvars
import uuid
import hashlib
import json

import graphsignal
from graphsignal import version
from graphsignal.proto import signals_pb2

logger = logging.getLogger('graphsignal')


span_stack_var = contextvars.ContextVar('span_stack_var', default=[])


def clear_span_stack():
    span_stack_var.set([])


def push_current_span(span):
    span_stack_var.set(span_stack_var.get() + [span])


def pop_current_span(span):
    span_stack = span_stack_var.get()
    if len(span_stack) > 0 and span_stack[-1] == span:
        span_stack_var.set(span_stack[:-1])
        return span_stack[-1]
    return None


def get_root_span():
    span_stack = span_stack_var.get()
    if len(span_stack) > 0:
        return span_stack[0]
    return None


def get_parent_span():
    span_stack = span_stack_var.get()
    if len(span_stack) > 1:
        return span_stack[-2]
    return None


def get_current_span():
    span_stack = span_stack_var.get()
    if len(span_stack) > 0:
        return span_stack[-1]
    return None


def _tracer():
    return graphsignal._tracer


class Payload:
    __slots__ = [
        'name',
        'content',
        'usage',
        'record_payload'
    ]

    def __init__(self, name, content, usage=None, record_payload=True):
        self.name = name
        self.content = content
        self.usage = usage
        self.record_payload = record_payload


class Usage:
    __slots__ = [
        'name',
        'value'
    ]

    def __init__(self, name, value):
        self.name = name
        self.value = value


class Span:
    MAX_RUN_TAGS = 10
    MAX_SPAN_TAGS = 10
    MAX_PAYLOADS = 10
    MAX_PAYLOAD_BYTES = 256 * 1024
    MAX_USAGES_COUNTERS = 10

    __slots__ = [
        '_operation',
        '_tags',
        '_root_span',
        '_parent_span',
        '_is_root',
        '_context',
        '_proto',
        '_is_started',
        '_is_stopped',
        '_start_counter',
        '_stop_counter',
        '_first_token_counter',
        '_exc_infos',
        '_payloads',
        '_usage'
    ]

    def __init__(self, operation, tags=None):
        push_current_span(self)

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

        self._operation = _sanitize_str(operation)
        if tags is not None:
            self._tags = dict(tags)
        else:
            self._tags = None
        self._is_stopped = False
        self._root_span = None
        self._parent_span = None
        self._is_root = False
        self._start_counter = None
        self._stop_counter = None
        self._first_token_counter = None
        self._context = False
        self._proto = None
        self._exc_infos = None
        self._payloads = None
        self._usage = None

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

        self._parent_span = get_parent_span()
        self._is_root = not self._parent_span
        self._root_span = self._parent_span.root_span() if self._parent_span else self

        self._proto = signals_pb2.Span()
        self._proto.span_id = _uuid_sha1(size=12)

        entry = self._proto.config.add()
        entry.key = 'graphsignal.library.version'
        entry.value = version.__version__

        self._context = {}

        # emit start event
        try:
            _tracer().emit_span_start(self._proto, self._context)
        except Exception as exc:
            logger.error('Error in span start event handlers', exc_info=True)

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
        first_token_ns = None
        if self._first_token_counter:
            first_token_ns = self._first_token_counter - self._start_counter

        now = time.time()
        end_us = int(now * 1e6)
        start_us = int(end_us - latency_ns / 1e3)
        now = int(now)

        # emit stop event
        try:
            _tracer().emit_span_stop(self._proto, self._context)
        except Exception as exc:
            logger.error('Error in span stop event handlers', exc_info=True)

        span_tags = self._span_tags()

        # update RED metrics
        _tracer().metric_store().update_histogram(
            scope='performance', name='latency', tags=span_tags, value=latency_ns, update_ts=now, is_time=True)
        if first_token_ns:
            _tracer().metric_store().update_histogram(
                scope='performance', name='first_token', tags=span_tags, value=first_token_ns, update_ts=now, is_time=True)
        _tracer().metric_store().inc_counter(
            scope='performance', name='call_count', tags=span_tags, value=1, update_ts=now)
        if self._exc_infos and len(self._exc_infos) > 0:
            for exc_info in self._exc_infos:
                if exc_info[0] is not None:
                    _tracer().metric_store().inc_counter(
                        scope='performance', name='exception_count', tags=span_tags, value=1, update_ts=now)
                    self.set_tag('exception', exc_info[0].__name__)

        # update payload usage metrics
        if self._payloads is not None:
            for payload in self._payloads.values():
                usage_tags = span_tags.copy()
                usage_tags['payload'] = payload.name
                if payload.usage:
                    for name, value in payload.usage.items():
                        _tracer().metric_store().inc_counter(
                            scope='data', name=name, tags=usage_tags, value=value, update_ts=now)

        if self._usage is not None:
            for usage in self._usage.values():
                usage_tags = span_tags.copy()
                _tracer().metric_store().inc_counter(
                    scope='data', name=usage.name, tags=usage_tags, value=usage.value, update_ts=now)

        # emit read event
        try:
            _tracer().emit_span_read(self._proto, self._context)
        except Exception as exc:
            logger.error('Error in span read event handlers', exc_info=True)

        # update recorder metrics
        if _tracer().check_metric_read_interval(now):
            _tracer().set_metric_read(now)
            try:
                _tracer().emit_metric_update()
            except Exception as exc:
                logger.error('Error in span read event handlers', exc_info=True)

        # fill and upload span
        # copy data to span proto
        self._proto.start_us = start_us
        self._proto.end_us = end_us

        # copy span context
        self._proto.context.start_ns = self._start_counter
        self._proto.context.end_ns = self._stop_counter
        if self._first_token_counter:
            self._proto.context.first_token_ns = self._first_token_counter
        if self._parent_span and self._parent_span._proto:
            self._proto.context.parent_span_id = self._parent_span._proto.span_id
        if self._root_span and self._root_span._proto:
            self._proto.context.root_span_id = self._root_span._proto.span_id

        # copy tags
        for key, value in span_tags.items():
            tag = self._proto.tags.add()
            tag.key = _sanitize_str(key, max_len=50)
            tag.value = _sanitize_str(value, max_len=250)

        # copy exception
        if self._exc_infos:
            for exc_info in self._exc_infos:
                exception_proto = self._proto.exceptions.add()
                if exc_info[0] and hasattr(exc_info[0], '__name__'):
                    exception_proto.exc_type = str(exc_info[0].__name__)
                if exc_info[1]:
                    exception_proto.message = str(exc_info[1])
                if exc_info[2]:
                    frames = traceback.format_tb(exc_info[2])
                    if len(frames) > 0:
                        exception_proto.stack_trace = ''.join(frames)

        # copy usage counters
        if self._payloads is not None:
            for payload in self._payloads.values():
                if payload.usage:
                    for name, value in payload.usage.items():
                        uc = self._proto.usage.add()
                        uc.payload_name = payload.name
                        uc.name = name
                        uc.value = value
        if self._usage is not None:
            for usage in self._usage.values():
                uc = self._proto.usage.add()
                uc.name = usage.name
                uc.value = usage.value


        # copy data payload
        if self._payloads is not None:
            for payload in self._payloads.values():
                if _tracer().record_payloads and payload.record_payload:
                    try:
                        content_type, content_bytes = encode_data_payload(payload.content)
                        if len(content_bytes) <= Span.MAX_PAYLOAD_BYTES:
                            payload_proto = self._proto.payloads.add()
                            payload_proto.name = payload.name
                            payload_proto.content_type = content_type
                            payload_proto.content_bytes = content_bytes
                    except Exception as exc:
                        logger.debug('Error encoding {0} payload for operation {1}'.format(payload.name, self._operation))

        # queue span proto for upload
        _tracer().uploader().upload_span(self._proto)

        # trigger upload
        if self._is_root:
            _tracer().tick(now)

    def measure(self) -> None:
        if not self._is_stopped:
            self._measure()

    def first_token(self) -> None:
        if not self._first_token_counter:
            self._first_token_counter = time.perf_counter_ns()

    def stop(self) -> None:
        try:
            self._stop()
        except Exception:
            logger.error('Error stopping span', exc_info=True)
        finally:
            self._is_stopped = True
            pop_current_span(self)

    def root_span(self) -> Optional['Span']:
        return self._root_span

    def parent_span(self) -> Optional['Span']:
        return self._parent_span

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

    def set_payload(
            self, 
            name: str, 
            content: Any, 
            usage: Optional[Dict[str, Union[int, float]]] = None, 
            record_payload: Optional[bool] = True) -> None:
        if self._payloads is None:
            self._payloads = {}

        if not name or not isinstance(name, str):
            logger.error('set_payload: name must be string')
            return

        if usage and not isinstance(usage, dict):
            logger.error('set_payload: usage must be dict')
            return

        if len(self._payloads) > Span.MAX_PAYLOADS:
            logger.error('set_payload: too many payloads (>{0})'.format(Span.MAX_PAYLOADS))
            return

        self._payloads[name] = Payload(
            name=name,
            content=content,
            usage=usage,
            record_payload=record_payload)

    def append_payload(
            self, 
            name: str, 
            content: Any, 
            usage: Optional[Dict[str, float]] = None, 
            record_payload: Optional[bool] = True) -> None:
        if self._payloads is None:
            self._payloads = {}

        if not name or not isinstance(name, str):
            logger.error('append_payload: name must be string')
            return

        if usage and not isinstance(usage, dict):
            logger.error('append_payload: usage must be dict')
            return

        if len(self._payloads) > Span.MAX_PAYLOADS:
            logger.error('append_payload: too many payloads (>{0})'.format(Span.MAX_PAYLOADS))
            return

        if name in self._payloads:
            payload = self._payloads[name]
            payload.content += content
            if usage:
                if payload.usage is None:
                    payload.usage = {}
                for name, value in usage.items():
                    if name in payload.usage:
                        payload.usage[name] += value
                    else:
                        payload.usage[name] = value
        else:
            self._payloads[name] = Payload(
                name=name,
                content=content,
                usage=usage,
                record_payload=record_payload)

    def set_usage(self, name: str, value: int) -> None:
        if self._usage is None:
            self._usage = {}

        if name and not isinstance(name, str):
            logger.error('set_usage: name must be string')
            return

        if value and not isinstance(value, (int, float)):
            logger.error('set_usage: value must be number')
            return

        if len(self._usage) > Span.MAX_USAGES_COUNTERS:
            logger.error('set_usage: too many usage counters (>{0})'.format(Span.MAX_USAGES_COUNTERS))
            return

        self._usage[name] = Usage(name=name, value=value)

    def score(
            self,
            name: str, 
            score: Optional[Union[int, float]] = None, 
            severity: Optional[int] = None,
            comment: Optional[str] = None) -> None:
        now = int(time.time())

        if not name:
            logger.error('Span.score: name is required')
            return

        score_obj = signals_pb2.Score()
        score_obj.score_id = _uuid_sha1(size=12)
        score_obj.span_id = self._proto.span_id
        score_obj.name = name

        for tag_key, tag_value in self._span_tags().items():
            tag = score_obj.tags.add()
            tag.key = _sanitize_str(tag_key, max_len=50)
            tag.value = _sanitize_str(tag_value, max_len=250)

        if score is not None:
            score_obj.score = score

        if severity and severity >= 1 and severity <= 5:
            score_obj.severity = severity

        if comment:
            score_obj.comment = comment
        
        score_obj.create_ts = now

        _tracer().uploader().upload_score(score_obj)
        _tracer().tick(now)


    def _span_tags(self, extra_tags=None):
        span_tags = {
            'deployment': _tracer().deployment, 
            'operation': self._operation}
        if _tracer().hostname:
            span_tags['hostname'] = _tracer().hostname
        if _tracer().tags is not None:
            span_tags.update(_tracer().tags)
        context_tags = _tracer().context_tags.get()
        if len(context_tags) > 0:
            span_tags.update(context_tags)
        if self._tags is not None:
            span_tags.update(self._tags)
        if extra_tags is not None:
            span_tags.update(extra_tags)
        return span_tags

    def repr(self):
        return 'Span({0})'.format(self._operation)


def _sanitize_str(val, max_len=250):
    if not isinstance(val, str):
        return str(val)[:max_len]
    else:
        return val[:max_len]


def _sha1(text, size=-1):
    sha1_hash = hashlib.sha1()
    sha1_hash.update(text.encode('utf-8'))
    return sha1_hash.hexdigest()[0:size]


def _uuid_sha1(size=-1):
    return _sha1(str(uuid.uuid4()), size)


def encode_data_payload(data):
    data_dict = _obj_to_dict(data)
    return ('application/json', json.dumps(data_dict).encode('utf-8'))


def _obj_to_dict(obj, level=0):
    if level >= 10:
        return
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    elif isinstance(obj, dict):
        return {k: _obj_to_dict(v, level=level+1) for k, v in obj.items()}
    elif isinstance(obj, (list, set, tuple)):
        return [_obj_to_dict(e, level=level+1) for e in obj]
    elif hasattr(obj, '__dict__'):
        return _obj_to_dict(vars(obj), level=level+1)
    else:
        return str(obj)