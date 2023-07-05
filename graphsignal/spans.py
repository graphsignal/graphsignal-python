from typing import Union, Any, Optional, Dict
import logging
import sys
import time
import traceback
import contextvars
import uuid
import hashlib

import graphsignal
from graphsignal import version
from graphsignal.proto import signals_pb2
from graphsignal.data import compute_data_stats, encode_data_sample
from graphsignal.proto_utils import parse_semver

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


class TraceOptions:
    __slots__ = [
        'record_samples',
        'record_metrics',
        'enable_profiling'
    ]

    def __init__(self, 
            record_samples: bool = True,
            record_metrics: bool = True,
            enable_profiling: bool = False):
        self.record_samples = record_samples
        self.record_metrics = record_metrics
        self.enable_profiling = enable_profiling

DEFAULT_OPTIONS = TraceOptions()


class DataObject:
    __slots__ = [
        'name',
        'obj',
        'counts',
        'record_data_sample'
    ]

    def __init__(self, name, obj, counts=None, record_data_sample=True):
        self.name = name
        self.obj = obj
        self.counts = counts
        self.record_data_sample = record_data_sample


class Span:
    MAX_RUN_TAGS = 10
    MAX_SPAN_TAGS = 10
    MAX_RUN_PARAMS = 10
    MAX_TRACE_PARAMS = 10
    MAX_DATA_OBJECTS = 10
    MAX_SAMPLE_BYTES = 256 * 1024

    __slots__ = [
        '_operation',
        '_tags',
        '_params',
        '_options',
        '_root_span',
        '_parent_span',
        '_is_sampling',
        '_is_root',
        '_latency_ns',
        '_context',
        '_proto',
        '_is_started',
        '_is_stopped',
        '_start_counter',
        '_stop_counter',
        '_exc_infos',
        '_data_objects'
    ]

    def __init__(self, operation, tags=None, options=None):
        push_current_span(self)

        self._is_started = False

        if not operation:
            logger.error('operation is required')
            return
        if tags is not None:
            if not isinstance(tags, dict):
                logger.error('tags must be dict')
                return
            if len(tags) > Span.MAX_SPAN_TAGS:
                logger.error('too many tags (>{0})'.format(Span.MAX_SPAN_TAGS))
                return

        self._operation = _sanitize_str(operation)
        if tags is not None:
            self._tags = dict(tags)
        else:
            self._tags = None
        self._params = None
        if options is None:
            self._options = DEFAULT_OPTIONS
        else:
            self._options = options
        self._is_stopped = False
        self._root_span = None
        self._parent_span = None
        self._is_sampling = False
        self._is_root = False
        self._latency_ns = None
        self._start_counter = None
        self._stop_counter = None
        self._context = False
        self._proto = None
        self._exc_infos = None
        self._data_objects = None

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

    def _init_sampling(self, sampling_type):
        self._is_sampling = True
        self._proto = signals_pb2.Span()
        self._proto.span_id = _uuid_sha1(size=12)
        self._proto.sampling_type = sampling_type
        self._proto.tracer_info.tracer_type = signals_pb2.TracerInfo.TracerType.PYTHON_TRACER
        parse_semver(self._proto.tracer_info.version, version.__version__)
        self._context = {}

    def _propagate_sampling(self, level=0):
        if level > 10:
            return
        if self._parent_span:
            if not self._parent_span.is_sampling():
                self._parent_span._init_sampling(sampling_type=signals_pb2.Span.SamplingType.CHILD_SAMPLING)
                self._parent_span._propagate_sampling(level+1)

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

        if self._options.record_samples:
            # sample if parent span is being sampled or this span is root
            if self._is_root:
                if _tracer().random_sampler().sample():
                    self._init_sampling(sampling_type=signals_pb2.Span.SamplingType.RANDOM_SAMPLING)
            else:
                if self._parent_span.is_sampling():
                    self._init_sampling(sampling_type=signals_pb2.Span.SamplingType.PARENT_SAMPLING)

            # emit start event
            if self._is_sampling:
                try:
                    _tracer().emit_span_start(self._proto, self._context, self._options)
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
        self._latency_ns = self._stop_counter - self._start_counter

        now = time.time()
        end_us = int(now * 1e6)
        start_us = int(end_us - self._latency_ns / 1e3)
        now = int(now)

        # emit stop event
        if self._is_sampling:
            try:
                _tracer().emit_span_stop(self._proto, self._context, self._options)
            except Exception as exc:
                logger.error('Error in span stop event handlers', exc_info=True)

        span_tags = self._span_tags()

        # if exception, but the span is not being recorded, try to start tracing
        if self._options.record_samples:
            if not self._is_sampling and self._exc_infos and len(self._exc_infos) > 0:
                if _tracer().random_sampler().sample():
                    self._init_sampling(sampling_type=signals_pb2.Span.SamplingType.ERROR_SAMPLING)
                    self._propagate_sampling()

        # update RED metrics
        if self._options.record_metrics:
            _tracer().metric_store().update_histogram(
                scope='performance', name='latency', tags=span_tags, value=self._latency_ns, update_ts=now, is_time=True)
            _tracer().metric_store().inc_counter(
                scope='performance', name='call_count', tags=span_tags, value=1, update_ts=now)
            if self._exc_infos and len(self._exc_infos) > 0:
                for exc_info in self._exc_infos:
                    if exc_info[0] is not None:
                        _tracer().metric_store().inc_counter(
                            scope='performance', name='exception_count', tags=span_tags, value=1, update_ts=now)
                        self.set_tag('exception', exc_info[0].__name__)

        # compute data statistics
        data_stats = None
        if self._options.record_metrics and self._data_objects is not None:
            data_stats = {}
            for data_obj in self._data_objects.values():
                try:
                    stats = compute_data_stats(data_obj.obj)
                    if not stats:
                        continue
                    data_stats[data_obj.name] = stats

                    # add/update extra counts to computed counts
                    if data_obj.counts is not None:
                        stats.counts.update(data_obj.counts)

                    # update data metrics
                    data_tags = span_tags.copy()
                    data_tags['data'] = data_obj.name
                    for count_name, count in stats.counts.items():
                        _tracer().metric_store().inc_counter(
                            scope='data', name=count_name, tags=data_tags, value=count, update_ts=now)
                except Exception as exc:
                    logger.error('Error computing data stats', exc_info=True)

        # emit read event
        if self._is_sampling:
            try:
                _tracer().emit_span_read(self._proto, self._context, self._options)
            except Exception as exc:
                logger.error('Error in span read event handlers', exc_info=True)

        # update recorder metrics
        if self._options.record_metrics and _tracer().check_metric_read_interval(now):
            _tracer().set_metric_read(now)
            try:
                _tracer().emit_metric_update()
            except Exception as exc:
                logger.error('Error in span read event handlers', exc_info=True)

        # fill and upload span
        if self._is_sampling:
            # copy data to span proto
            self._proto.start_us = start_us
            self._proto.end_us = end_us
            if self._is_root:
                self._proto.labels.insert(0, 'root')
            self._proto.process_usage.start_ms = _tracer()._process_start_ms

            # copy span context
            self._proto.context.start_ns = self._start_counter
            self._proto.context.end_ns = self._stop_counter
            if self._parent_span and self._parent_span._proto:
                self._proto.context.parent_span_id = self._parent_span._proto.span_id
            if self._root_span and self._root_span._proto:
                self._proto.context.root_span_id = self._root_span._proto.span_id

            # copy tags
            for key, value in span_tags.items():
                tag = self._proto.tags.add()
                tag.key = _sanitize_str(key, max_len=50)
                tag.value = _sanitize_str(value, max_len=250)

            # copy params
            if self._params is not None:
                for name, value in self._params.items():
                    param = self._proto.params.add()
                    param.name = _sanitize_str(name, max_len=50)
                    param.value = _sanitize_str(value, max_len=250)

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

            # copy data stats
            if data_stats is not None:
                for data_name, stats in data_stats.items():
                    data_stats_proto = self._proto.data_profile.add()
                    data_stats_proto.data_name = data_name
                    if stats.type_name:
                        data_stats_proto.data_type = stats.type_name
                    if stats.shape:
                        data_stats_proto.shape[:] = stats.shape
                    for counter_name, count in stats.counts.items():
                        if count > 0:
                            dc = data_stats_proto.counts.add()
                            dc.name = counter_name
                            dc.count = count

                    # copy data sample
                    data_obj = self._data_objects[data_name]
                    if _tracer().record_data_samples and data_obj.record_data_sample:
                        try:
                            sample = encode_data_sample(data_obj.obj)
                            if sample is not None and len(sample.content_bytes) <= Span.MAX_SAMPLE_BYTES:
                                sample_proto = self._proto.data_samples.add()
                                sample_proto.data_name = data_name
                                sample_proto.content_type = sample.content_type
                                sample_proto.content_bytes = sample.content_bytes
                        except Exception as exc:
                            logger.debug('Error encoding {0} sample for operation {1}'.format(data_name, self._operation))
                    else:
                        data_stats_proto.sample_recording_disabled = True

            # queue span proto for upload
            _tracer().uploader().upload_span(self._proto)

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
            pop_current_span(self)

    def root_span(self) -> Optional['Span']:
        return self._root_span

    def parent_span(self) -> Optional['Span']:
        return self._parent_span

    def is_sampling(self):
        return self._is_sampling

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

    def set_param(self, name: str, value: str) -> None:
        if not name:
            logger.error('set_param: name must be provided')
            return

        if self._params is None:
            self._params = {}

        if len(self._params) > Span.MAX_TRACE_PARAMS:
            logger.error('set_param: too many params (>{0})'.format(Span.MAX_TRACE_PARAMS))
            return

        self._params[name] = value

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

    def set_data(self, 
            name: str, 
            obj: Any, 
            counts: Optional[Dict[str, int]] = None, 
            record_data_sample: Optional[bool] = True) -> None:
        if self._data_objects is None:
            self._data_objects = {}

        if name and not isinstance(name, str):
            logger.error('set_data: name must be string')
            return

        if counts and not isinstance(counts, dict):
            logger.error('append_data: name must be dict')
            return

        if len(self._data_objects) > Span.MAX_DATA_OBJECTS:
            logger.error('set_data: too many data objects (>{0})'.format(Span.MAX_DATA_OBJECTS))
            return

        self._data_objects[name] = DataObject(
            name=name,
            obj=obj,
            counts=counts,
            record_data_sample=record_data_sample)

    def append_data(self, 
            name: str, 
            obj: Any, 
            counts: Optional[Dict[str, int]] = None, 
            record_data_sample: Optional[bool] = True) -> None:
        if self._data_objects is None:
            self._data_objects = {}

        if name and not isinstance(name, str):
            logger.error('append_data: name must be string')
            return

        if counts and not isinstance(counts, dict):
            logger.error('append_data: name must be dict')
            return

        if len(self._data_objects) > Span.MAX_DATA_OBJECTS:
            logger.error('append_data: too many data objects (>{0})'.format(Span.MAX_DATA_OBJECTS))
            return

        if name in self._data_objects:
            data_obj = self._data_objects[name]
            data_obj.obj += obj
            if counts:
                if data_obj.counts is None:
                    data_obj.counts = {}
                for count_name, value in counts.items():
                    if count_name in data_obj.counts:
                        data_obj.counts[count_name] += value
                    else:
                        data_obj.counts[count_name] = value
        else:
            self._data_objects[name] = DataObject(
                name=name,
                obj=obj,
                counts=counts,
                record_data_sample=record_data_sample)

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
