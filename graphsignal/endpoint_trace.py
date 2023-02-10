from typing import Union, Any, Optional, Dict
import logging
import sys
import time
import traceback

import graphsignal
from graphsignal.proto import signals_pb2
from graphsignal.data import compute_data_stats
from graphsignal.span_context import start_root_span, start_span, stop_span

logger = logging.getLogger('graphsignal')

SAMPLE_TRACES = {1, 10, 100, 1000}


class TraceOptions:
    __slots__ = [
        'auto_sampling',
        'ensure_sample',
        'enable_profiling'
    ]

    def __init__(self, 
            auto_sampling: bool = True, 
            ensure_sample: bool = False, 
            enable_profiling: bool = False):
        self.auto_sampling = auto_sampling
        self.ensure_sample = ensure_sample
        self.enable_profiling = enable_profiling

DEFAULT_OPTIONS = TraceOptions()


class DataObject:
    __slots__ = [
        'name',
        'obj',
        'extra_counts',
        'check_missing_values'
    ]

    def __init__(self, name, obj, extra_counts=None, check_missing_values=False):
        self.name = name
        self.obj = obj
        self.extra_counts = extra_counts
        self.check_missing_values = check_missing_values


class EndpointTrace:
    MAX_RUN_TAGS = 10
    MAX_TRACE_TAGS = 10
    MAX_RUN_PARAMS = 10
    MAX_TRACE_PARAMS = 10
    MAX_DATA_OBJECTS = 10

    __slots__ = [
        '_trace_sampler',
        '_metric_store',
        '_mv_detector',
        '_agent',
        '_endpoint',
        '_tags',
        '_params',
        '_options',
        '_is_sampling',
        '_latency_us',
        '_context',
        '_signal',
        '_is_stopped',
        '_start_counter',
        '_stop_counter',
        '_exc_info',
        '_data_objects',
        '_root_span',
        '_has_missing_values'
    ]

    def __init__(self, endpoint, tags=None, options=None):
        if not endpoint:
            raise ValueError('endpoint is required')
        if not isinstance(endpoint, str):
            raise ValueError('endpoint must be string')
        if len(endpoint) > 50:
            raise ValueError('endpoint is too long (>50)')
        if tags is not None:
            if not isinstance(tags, dict):
                raise ValueError('tags must be dict')
            if len(tags) > EndpointTrace.MAX_TRACE_TAGS:
                raise ValueError('too many tags (>{0})'.format(EndpointTrace.MAX_TRACE_TAGS))

        self._endpoint = endpoint
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
        self._metric_store = None
        self._mv_detector = None
        self._is_stopped = False
        self._is_sampling = False
        self._latency_us = None
        self._start_counter = None
        self._stop_counter = None
        self._context = False
        self._signal = None
        self._exc_info = None
        self._data_objects = None
        self._root_span = None
        self._has_missing_values = False

        try:
            self._start()
        except Exception as ex:
            if self._is_sampling:
                self._is_stopped = True
                self._trace_sampler.unlock()
            raise ex

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        if exc_info and exc_info[1] and isinstance(exc_info[1], Exception):
            self._exc_info = exc_info
        self.stop()
        return False

    def _init_sampling(self):
        self._is_sampling = True
        self._signal = self._agent.create_signal()
        self._context = {}

    def _start(self):
        if self._is_stopped:
            return

        self._trace_sampler = self._agent.trace_sampler(self._endpoint)
        self._metric_store = self._agent.metric_store(self._endpoint)
        self._mv_detector = self._agent.mv_detector()

        lock_group = 'samples'
        if self._options.auto_sampling:
            lock_group += '-auto'
        if self._options.ensure_sample:
            lock_group += '-ensured'
        if self._options.enable_profiling:
            lock_group += '-profiled'

        if ((self._options.auto_sampling and self._trace_sampler.lock(lock_group, include_trace_idx=SAMPLE_TRACES)) or
                (self._options.ensure_sample and self._trace_sampler.lock(lock_group, limit_per_interval=2))):
            self._init_sampling()

            # emit start event
            try:
                self._agent.emit_trace_start(self._signal, self._context, self._options)
            except Exception as exc:
                logger.error('Error in trace start event handlers', exc_info=True)
                self._add_agent_exception(exc)

        self._start_counter = time.perf_counter_ns()

        # start current span
        if self._is_sampling:
            # root span represents the current trace
            # only recording nested spans of a trace that is being sampled
            self._root_span = start_root_span(name=self._endpoint, start_ns=self._start_counter, is_endpoint=True)
        else:
            start_span(name=self._endpoint, start_ns=self._start_counter, is_endpoint=True)

    def _measure(self) -> None:
        self._stop_counter = time.perf_counter_ns()

    def _stop(self) -> None:
        if self._is_stopped:
            return
        self._is_stopped = True

        if self._stop_counter is None:
            self._measure()
        self._latency_us = int((self._stop_counter - self._start_counter) / 1e3)
        end_us = _timestamp_us()

        # stop current span
        stop_span(end_ns=self._stop_counter, has_exception=bool(self._exc_info and self._exc_info[0]))

        if self._is_sampling:
            # emit stop event
            try:
                self._agent.emit_trace_stop(self._signal, self._context, self._options)
            except Exception as exc:
                logger.error('Error in trace stop event handlers', exc_info=True)
                self._add_agent_exception(exc)

        # if exception, but the trace is not being recorded, try to start tracing
        if not self._is_sampling and self._exc_info and self._exc_info[0]:
            if self._trace_sampler.lock('exceptions'):
                self._init_sampling()

        # update time and counters
        self._metric_store.add_time(self._latency_us)
        self._metric_store.inc_call_count(1, end_us)

        # compute data statistics
        data_stats = None
        if self._data_objects is not None:
            data_stats = {}
            for data_obj in self._data_objects.values():
                try:
                    stats = compute_data_stats(data_obj.obj)
                    if not stats:
                        continue
                    data_stats[data_obj.name] = stats

                    # add/update extra counts to computed counts
                    if data_obj.extra_counts is not None:
                        stats.counts.update(data_obj.extra_counts)

                    # check missing values
                    if data_obj.check_missing_values:
                        if self._mv_detector.detect(data_obj.name, stats.counts):
                            self._has_missing_values = True

                    # update data metrics
                    for count_name, count in stats.counts.items():
                        self._metric_store.inc_data_counter(
                            data_obj.name, count_name, count, end_us)
                except Exception as exc:
                    logger.error('Error computing data stats', exc_info=True)
                    self._add_agent_exception(exc)

        # if missing values detected, but the trace is not being recorded, try to start tracing
        if not self._is_sampling and self._has_missing_values:
            if self._trace_sampler.lock('missing-values'):
                self._init_sampling()

        # update exception counter
        if self._exc_info and self._exc_info[0]:
            self._metric_store.inc_exception_count(1, end_us)

        # fill and upload signal
        if self._is_sampling:
            # emit read event
            try:
                self._agent.emit_trace_read(self._signal, self._context, self._options)
            except Exception as exc:
                logger.error('Error in trace read event handlers', exc_info=True)
                self._add_agent_exception(exc)

            # copy data to signal
            self._signal.endpoint_name = self._endpoint
            self._signal.start_us = end_us - self._latency_us
            self._signal.end_us = end_us
            if self._exc_info and self._exc_info[0]:
                self._signal.signal_type = signals_pb2.SignalType.EXCEPTION_SIGNAL
            elif self._has_missing_values:
                self._signal.signal_type = signals_pb2.SignalType.MISSING_VALUES_SIGNAL
            else:
                self._signal.signal_type = signals_pb2.SignalType.SAMPLE_SIGNAL
            self._signal.process_usage.start_ms = self._agent._process_start_ms

            # copy tags
            tags = None
            if self._agent.tags is not None:
                tags = self._agent.tags.copy()
                if self._tags is not None:
                    tags.update(self._tags)
            elif self._tags is not None:
                tags = self._tags
            if tags is not None:
                for key, value in tags.items():
                    tag = self._signal.tags.add()
                    tag.key = key[:50]
                    tag.value = str(value)[:50]
                    if self._tags and key in self._tags:
                        tag.is_trace_level = True

            # copy params
            params = None
            if self._agent.params is not None:
                params = self._agent.params.copy()
                if self._params is not None:
                    params.update(self._params)
            elif self._params is not None:
                params = self._params
            if params is not None:
                for name, value in params.items():
                    param = self._signal.params.add()
                    param.name = name[:50]
                    param.value = str(value)[:50]
                    if self._params and name in self._params:
                        param.is_trace_level = True

            # copy metrics
            self._metric_store.convert_to_proto(self._signal, end_us)

            # copy trace measurements
            self._signal.trace_sample.trace_idx = self._trace_sampler.current_trace_idx()
            self._signal.trace_sample.latency_us = self._latency_us
            self._signal.trace_sample.is_ensured = self._options.ensure_sample
            self._signal.trace_sample.is_profiled = self._options.enable_profiling

            # copy exception
            if self._exc_info and self._exc_info[0]:
                exception = self._signal.exceptions.add()
                if self._exc_info[0] and hasattr(self._exc_info[0], '__name__'):
                    exception.exc_type = str(self._exc_info[0].__name__)
                if self._exc_info[1]:
                    exception.message = str(self._exc_info[1])
                if self._exc_info[2]:
                    frames = traceback.format_tb(self._exc_info[2])
                    if len(frames) > 0:
                        exception.stack_trace = ''.join(frames)

            # copy data counts
            if data_stats is not None:
                for name, stats in data_stats.items():
                    data_stats_proto = self._signal.data_profile.add()
                    data_stats_proto.data_name = name
                    if stats.type_name:
                        data_stats_proto.data_type = stats.type_name
                    if stats.shape:
                        data_stats_proto.shape[:] = stats.shape
                    for name, count in stats.counts.items():
                        if count > 0:
                            dc = data_stats_proto.counts.add()
                            dc.name = name
                            dc.count = count

            # copy spans
            if self._root_span:
                _convert_span_to_proto(self._signal.root_span, self._root_span)

            # queue signal for upload
            self._agent.uploader().upload_signal(self._signal)
            self._agent.tick()

    def measure(self) -> None:
        if not self._is_stopped:
            self._measure()

    def stop(self) -> None:
        try:
            self._stop()
        finally:
            if self._is_sampling:
                self._is_stopped = True
                self._trace_sampler.unlock()

    def is_sampling(self):
        return self._is_sampling

    def set_tag(self, key: str, value: str) -> None:
        if not key:
            raise ValueError('set_tag: key must be provided')
        if value is None:
            raise ValueError('set_tag: value must be provided')

        if self._tags is None:
            self._tags = {}

        if len(self._tags) > EndpointTrace.MAX_TRACE_TAGS:
            raise ValueError('set_tag: too many tags (>{0})'.format(EndpointTrace.MAX_TRACE_TAGS))

        self._tags[key] = value

    def set_param(self, name: str, value: str) -> None:
        if not name:
            raise ValueError('set_param: name must be provided')
        if value is None:
            raise ValueError('set_param: value must be provided')

        if self._params is None:
            self._params = {}

        if len(self._params) > EndpointTrace.MAX_TRACE_PARAMS:
            raise ValueError('set_param: too many params (>{0})'.format(EndpointTrace.MAX_TRACE_PARAMS))

        self._params[name] = value

    def set_exception(self, exc: Optional[Exception] = None, exc_info: Optional[bool] = None) -> None:
        if exc is not None and not isinstance(exc, Exception):
            raise ValueError('set_exception: exc must be instance of Exception')

        if exc_info is not None and not isinstance(exc_info, bool):
            raise ValueError('set_exception: exc_info must be bool')

        if exc:
            self._exc_info = (exc.__class__, str(exc), exc.__traceback__)
        elif exc_info == True:
            self._exc_info = sys.exc_info()

    def set_data(self, 
            name: str, 
            obj: Any, 
            extra_counts: Optional[Dict[str, int]] = None, 
            check_missing_values: Optional[bool] = False) -> None:
        if self._data_objects is None:
            self._data_objects = {}

        if name and not isinstance(name, str):
            raise ValueError('set_data: name must be string')

        if len(self._data_objects) > EndpointTrace.MAX_DATA_OBJECTS:
            raise ValueError('set_data: too many data objects (>{0})'.format(EndpointTrace.MAX_DATA_OBJECTS))

        self._data_objects[name] = DataObject(
            name=name, obj=obj, extra_counts=extra_counts, check_missing_values=check_missing_values)

    def get_latency_us(self):
        return self._latency_us

    def _add_agent_exception(self, exc):
        if not self._is_sampling:
            return

        agent_error = self._signal.agent_errors.add()
        agent_error.message = str(exc)
        if exc.__traceback__:
            frames = traceback.format_tb(exc.__traceback__)
            if len(frames) > 0:
                agent_error.stack_trace = ''.join(frames)

    def repr(self):
        return 'EndpointTrace({0})'.format(self._endpoint)


def _timestamp_us():
    return int(time.time() * 1e6)


def _convert_span_to_proto(proto, span):
    proto.name = span.name
    proto.start_ns = span.start_ns
    proto.end_ns = span.end_ns
    proto.has_exception = span.has_exception
    proto.is_endpoint = span.is_endpoint
    if span.children is not None:
        for child in span.children:
            _convert_span_to_proto(proto.spans.add(), child)
