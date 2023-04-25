from typing import Union, Any, Optional, Dict
import logging
import sys
import time
import traceback

import graphsignal
from graphsignal.proto import signals_pb2
from graphsignal.data import compute_data_stats, encode_data_sample
from graphsignal.spans import get_current_span, Span

logger = logging.getLogger('graphsignal')


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
        'check_missing_values'
    ]

    def __init__(self, name, obj, counts=None, check_missing_values=False):
        self.name = name
        self.obj = obj
        self.counts = counts
        self.check_missing_values = check_missing_values


class Trace:
    MAX_RUN_TAGS = 10
    MAX_TRACE_TAGS = 10
    MAX_RUN_PARAMS = 10
    MAX_TRACE_PARAMS = 10
    MAX_DATA_OBJECTS = 10
    MAX_SAMPLE_BYTES = 32 * 1024
    MAX_NESTED_TRACES = 250

    __slots__ = [
        '_trace_sampler',
        '_lo_detector',
        '_mv_detector',
        '_agent',
        '_operation',
        '_tags',
        '_params',
        '_options',
        '_span',
        '_is_sampling',
        '_is_root',
        '_latency_ns',
        '_context',
        '_proto',
        '_is_started',
        '_is_stopped',
        '_start_counter',
        '_stop_counter',
        '_exc_info',
        '_data_objects',
        '_is_latency_outlier',
        '_has_missing_values'
    ]

    def __init__(self, operation, tags=None, options=None):
        self._is_started = False

        if not operation:
            logger.error('operation is required')
            return
        if tags is not None:
            if not isinstance(tags, dict):
                logger.error('tags must be dict')
                return
            if len(tags) > Trace.MAX_TRACE_TAGS:
                logger.error('too many tags (>{0})'.format(Trace.MAX_TRACE_TAGS))
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
        self._agent = graphsignal._agent
        self._trace_sampler = None
        self._lo_detector = None
        self._mv_detector = None
        self._is_stopped = False
        self._span = None
        self._is_sampling = False
        self._is_root = False
        self._latency_ns = None
        self._start_counter = None
        self._stop_counter = None
        self._context = False
        self._proto = None
        self._exc_info = None
        self._data_objects = None
        self._is_latency_outlier = False
        self._has_missing_values = False

        try:
            self._start()
        except Exception:
            logger.error('Error starting trace', exc_info=True)
            self._is_stopped = True

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        if exc_info and exc_info[1] and isinstance(exc_info[1], Exception):
            self._exc_info = exc_info
        self.stop()
        return False

    def _init_sampling(self, sampling_type):
        self._is_sampling = True
        self._proto = self._agent.create_trace_proto()
        self._proto.sampling_type = sampling_type
        self._context = {}
        self._span.set_trace_id(self._proto.trace_id)
        self._span.set_sampling(True)

    def _start(self):
        if self._is_started:
            return
        if self._is_stopped:
            return

        if self._agent.debug_mode:
            logger.debug(f'Starting trace {self._operation}')

        self._trace_sampler = self._agent.trace_sampler(self._operation)
        self._lo_detector = self._agent.lo_detector(self._operation)
        self._mv_detector = self._agent.mv_detector()

        parent_span = get_current_span()
        self._span = Span(self._operation)
        self._is_root = not parent_span

        if self._options.record_samples:
            # sample if parent trace is being sampled or this trace is root
            if self._is_root:
                if self._trace_sampler.sample('random'):
                    self._init_sampling(sampling_type=signals_pb2.Trace.SamplingType.RANDOM_SAMPLING)
            else:
                if parent_span.is_root_sampling() and parent_span.can_add_child():
                    self._init_sampling(sampling_type=signals_pb2.Trace.SamplingType.PARENT_SAMPLING)

            # emit start event
            if self._is_sampling:
                try:
                    self._agent.emit_trace_start(self._proto, self._context, self._options)
                except Exception as exc:
                    logger.error('Error in trace start event handlers', exc_info=True)
                    self._add_tracer_exception(exc)

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

        if self._agent.debug_mode:
            logger.debug(f'Stopping trace {self._operation}')

        if self._stop_counter is None:
            self._measure()
        self._latency_ns = self._stop_counter - self._start_counter

        now = time.time()
        end_us = int(now * 1e6)
        start_us = int(end_us - self._latency_ns / 1e3)
        now = int(now)

        # stop current span
        self._span.stop()

        # emit stop event
        if self._is_sampling:
            try:
                self._agent.emit_trace_stop(self._proto, self._context, self._options)
            except Exception as exc:
                logger.error('Error in trace stop event handlers', exc_info=True)
                self._add_tracer_exception(exc)

        trace_tags = self._trace_tags()

        # if exception, but the trace is not being recorded, try to start tracing
        if self._options.record_samples:
            if not self._is_sampling and self._exc_info and self._exc_info[0]:
                if self._trace_sampler.sample('exceptions'):
                    self._init_sampling(sampling_type=signals_pb2.Trace.SamplingType.ERROR_SAMPLING)

        # check for outliers
        if self._options.record_samples:
            self._is_latency_outlier = self._lo_detector.detect(self._latency_ns / 1e9)
            if not self._is_sampling and self._is_latency_outlier:
                if self._trace_sampler.sample('latency-outliers'):
                    self._init_sampling(sampling_type=signals_pb2.Trace.SamplingType.ERROR_SAMPLING)
        self._lo_detector.update(self._latency_ns / 1e9)

        # update RED metrics
        if self._options.record_metrics:
            self._agent.metric_store().update_histogram(
                scope='performance', name='latency', tags=trace_tags, value=self._latency_ns, update_ts=now, is_time=True)
            self._agent.metric_store().inc_counter(
                scope='performance', name='call_count', tags=trace_tags, value=1, update_ts=now)
            if self._exc_info and self._exc_info[0]:
                self._agent.metric_store().inc_counter(
                    scope='performance', name='exception_count', tags=trace_tags, value=1, update_ts=now)
                self.set_tag('exception', self._exc_info[0].__name__)

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

                    # check missing values
                    if data_obj.check_missing_values:
                        if self._mv_detector.detect(data_obj.name, stats.counts):
                            self._has_missing_values = True

                    # update data metrics
                    data_tags = trace_tags.copy()
                    data_tags['data'] = data_obj.name
                    for count_name, count in stats.counts.items():
                        self._agent.metric_store().inc_counter(
                            scope='data', name=count_name, tags=data_tags, value=count, update_ts=now)
                except Exception as exc:
                    logger.error('Error computing data stats', exc_info=True)
                    self._add_tracer_exception(exc)

        # if missing values detected, but the trace is not being recorded, try to start tracing
        if self._options.record_samples:
            if not self._is_sampling and self._has_missing_values:
                if self._trace_sampler.sample('missing-values'):
                    self._init_sampling(sampling_type=signals_pb2.Trace.SamplingType.ERROR_SAMPLING)

        # emit read event
        if self._is_sampling:
            try:
                self._agent.emit_trace_read(self._proto, self._context, self._options)
            except Exception as exc:
                logger.error('Error in trace read event handlers', exc_info=True)
                self._add_tracer_exception(exc)

        # update recorder metrics
        if self._options.record_metrics and self._agent.check_metric_read_interval(now):
            try:
                self._agent.emit_metric_update()
                self._agent.set_metric_read(now)
            except Exception as exc:
                logger.error('Error in trace read event handlers', exc_info=True)
                self._add_tracer_exception(exc)

        # fill and upload trace
        if self._is_sampling:
            # copy data to trace proto
            self._proto.start_us = start_us
            self._proto.end_us = end_us
            if self._is_root:
                self._proto.labels.insert(0, 'root')
            if self._exc_info and self._exc_info[0]:
                self._proto.labels.append('exception')
            if self._is_latency_outlier:
                self._proto.labels.append('latency-outlier')
            if self._has_missing_values:
                self._proto.labels.append('missing-values')
            self._proto.process_usage.start_ms = self._agent._process_start_ms

            # copy span
            self._proto.span.start_ns = self._start_counter
            self._proto.span.end_ns = self._stop_counter
            if self._span.parent_span and self._span.parent_span.trace_id:
                self._proto.span.parent_trace_id = self._span.parent_span.trace_id
            if self._span.root_span and self._span.root_span.trace_id:
                self._proto.span.root_trace_id = self._span.root_span.trace_id

            # copy tags
            for key, value in trace_tags.items():
                tag = self._proto.tags.add()
                tag.key = _sanitize_str(key)
                tag.value = _sanitize_str(value)

            # copy params
            if self._params is not None:
                for name, value in self._params.items():
                    param = self._proto.params.add()
                    param.name = _sanitize_str(name)
                    param.value = _sanitize_str(value)

            # copy exception
            if self._exc_info and self._exc_info[0]:
                exception_proto = self._proto.exceptions.add()
                if self._exc_info[0] and hasattr(self._exc_info[0], '__name__'):
                    exception_proto.exc_type = str(self._exc_info[0].__name__)
                if self._exc_info[1]:
                    exception_proto.message = str(self._exc_info[1])
                if self._exc_info[2]:
                    frames = traceback.format_tb(self._exc_info[2])
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
                    if self._agent.record_data_samples:
                        try:
                            sample = encode_data_sample(self._data_objects[data_name].obj)
                            if sample is not None and len(sample.content_bytes) <= Trace.MAX_SAMPLE_BYTES:
                                sample_proto = self._proto.data_samples.add()
                                sample_proto.data_name = data_name
                                sample_proto.content_type = sample.content_type
                                sample_proto.content_bytes = sample.content_bytes
                        except Exception as exc:
                            logger.error('Error encoding data sample', exc_info=True)
                            self._add_tracer_exception(exc)

            # queue trace proto for upload
            self._agent.uploader().upload_trace(self._proto)

        # trigger upload
        if self._is_root:
            self._agent.tick(now)

    def measure(self) -> None:
        if not self._is_stopped:
            self._measure()

    def stop(self) -> None:
        try:
            self._stop()
        except Exception:
            logger.error('Error stopping trace', exc_info=True)
        finally:
            self._is_stopped = True

    def is_sampling(self):
        return self._is_sampling

    def set_tag(self, key: str, value: str) -> None:
        if not key:
            logger.error('set_tag: key must be provided')
            return
        if value is None:
            logger.error('set_tag: value must be provided')
            return

        if self._tags is None:
            self._tags = {}

        if len(self._tags) > Trace.MAX_TRACE_TAGS:
            logger.error('set_tag: too many tags (>{0})'.format(Trace.MAX_TRACE_TAGS))
            return

        self._tags[key] = value

    def set_param(self, name: str, value: str) -> None:
        if not name:
            logger.error('set_param: name must be provided')
            return

        if self._params is None:
            self._params = {}

        if len(self._params) > Trace.MAX_TRACE_PARAMS:
            logger.error('set_param: too many params (>{0})'.format(Trace.MAX_TRACE_PARAMS))
            return

        self._params[name] = value

    def set_exception(self, exc: Optional[Exception] = None, exc_info: Optional[bool] = None) -> None:
        if exc is not None and not isinstance(exc, Exception):
            logger.error('set_exception: exc must be instance of Exception')
            return

        if exc_info is not None and not isinstance(exc_info, bool):
            logger.error('set_exception: exc_info must be bool')
            return

        if exc:
            self._exc_info = (exc.__class__, str(exc), exc.__traceback__)
        elif exc_info == True:
            self._exc_info = sys.exc_info()

    def set_data(self, 
            name: str, 
            obj: Any, 
            counts: Optional[Dict[str, int]] = None, 
            check_missing_values: Optional[bool] = False) -> None:
        if self._data_objects is None:
            self._data_objects = {}

        if name and not isinstance(name, str):
            logger.error('set_data: name must be string')
            return

        if counts and not isinstance(counts, dict):
            logger.error('append_data: name must be dict')
            return

        if len(self._data_objects) > Trace.MAX_DATA_OBJECTS:
            logger.error('set_data: too many data objects (>{0})'.format(Trace.MAX_DATA_OBJECTS))
            return

        self._data_objects[name] = DataObject(
            name=name, obj=obj, counts=counts, check_missing_values=check_missing_values)

    def append_data(self, 
            name: str, 
            obj: Any, 
            counts: Optional[Dict[str, int]] = None, 
            check_missing_values: Optional[bool] = False) -> None:
        if self._data_objects is None:
            self._data_objects = {}

        if name and not isinstance(name, str):
            logger.error('append_data: name must be string')
            return

        if counts and not isinstance(counts, dict):
            logger.error('append_data: name must be dict')
            return

        if len(self._data_objects) > Trace.MAX_DATA_OBJECTS:
            logger.error('append_data: too many data objects (>{0})'.format(Trace.MAX_DATA_OBJECTS))
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
                name=name, obj=obj, counts=counts, check_missing_values=check_missing_values)

    def _add_tracer_exception(self, exc):
        if not self._is_sampling:
            return

        tracer_error = self._proto.tracer_errors.add()
        tracer_error.message = str(exc)
        if exc.__traceback__:
            frames = traceback.format_tb(exc.__traceback__)
            if len(frames) > 0:
                tracer_error.stack_trace = ''.join(frames)

    def _trace_tags(self, extra_tags=None):
        trace_tags = {
            'deployment': self._agent.deployment, 
            'operation': self._operation}
        if self._agent.hostname:
            trace_tags['hostname'] = self._agent.hostname
        if self._agent.tags is not None:
            trace_tags.update(self._agent.tags)
        context_tags = self._agent.context_tags.get()
        if len(context_tags) > 0:
            trace_tags.update(context_tags)
        if self._tags is not None:
            trace_tags.update(self._tags)
        if extra_tags is not None:
            trace_tags.update(extra_tags)
        return trace_tags

    def repr(self):
        return 'Trace({0})'.format(self._operation)


def _sanitize_str(val, max_len=50):
    return str(val)[:max_len]