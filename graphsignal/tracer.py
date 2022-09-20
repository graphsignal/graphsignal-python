from typing import Union, Any, Optional, Dict
import logging
import sys
import time
import traceback

import graphsignal
from graphsignal.proto import signals_pb2
from graphsignal.data import compute_counts, build_stats 

logger = logging.getLogger('graphsignal')


class TraceSpan:
    MAX_TAGS = 10
    MAX_DATA_OBJECTS = 10

    __slots__ = [
        '_operation_profiler',
        '_trace_sampler',
        '_metric_store',
        '_agent',
        '_endpoint',
        '_ensure_trace',
        '_tags',
        '_is_tracing',
        '_is_profiling',
        '_signal',
        '_is_stopped',
        '_start_counter',
        '_exc_info',
        '_data'
    ]

    def __init__(self, 
            endpoint,
            tags=None,
            ensure_trace=False, 
            operation_profiler=None):
        if not endpoint:
            raise ValueError('endpoint is required')
        if not isinstance(endpoint, str):
            raise ValueError('endpoint must be string')
        if len(endpoint) > 50:
            raise ValueError('endpoint is too long (>50)')
        if tags is not None:
            if not isinstance(tags, dict):
                raise ValueError('tags must be dict')
            if len(tags) > TraceSpan.MAX_TAGS:
                raise ValueError('too many tags (>{0})'.format(TraceSpan.MAX_TAGS))

        self._endpoint = endpoint
        self._tags = tags
        self._ensure_trace = ensure_trace
        self._operation_profiler = operation_profiler

        self._agent = None
        self._trace_sampler = None
        self._metric_store = None
        self._is_stopped = False
        self._is_tracing = False
        self._is_profiling = False
        self._signal = None
        self._exc_info = None
        self._data = None

        try:
            self._start()
        except Exception as ex:
            if self._is_tracing:
                self._is_stopped = True
                self._trace_sampler.unlock()
            raise ex

    def is_tracing(self):
        return self._is_tracing

    def is_profiling(self):
        return self._is_profiling

    def _start(self):
        if self._is_stopped:
            return

        self._agent = graphsignal._agent
        self._trace_sampler = self._agent.get_trace_sampler(self._endpoint)
        self._metric_store = self._agent.get_metric_store(self._endpoint)

        if self._trace_sampler.lock(ensure=self._ensure_trace):
            if logger.isEnabledFor(logging.DEBUG):
                profiling_start_overhead_counter = time.perf_counter()

            self._is_tracing = True
            self._signal = self._agent.create_signal()

            # read framework-specific info
            if self._operation_profiler:
                self._operation_profiler.read_info(self._signal)

            # start profiler
            if self._operation_profiler and self._trace_sampler.should_profile():
                try:
                    self._operation_profiler.start(self._signal)
                    self._is_profiling = True
                except Exception as exc:
                    self._add_profiler_exception(exc)
                    logger.error('Error starting profiler', exc_info=True)

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('Profiling start took: %fs', time.perf_counter() - profiling_start_overhead_counter)

        self._start_counter = time.perf_counter()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self._exc_info = exc_info
        self.stop()
        return False

    def stop(self) -> None:
        try:
            self._stop()
        finally:
            if self._is_tracing:
                self._is_stopped = True
                self._trace_sampler.unlock()            

    def _stop(self) -> None:
        stop_counter = time.perf_counter()
        duration_us = int((stop_counter - self._start_counter) * 1e6)
        end_us = _timestamp_us()

        if self._is_stopped:
            return
        self._is_stopped = True

        # if exception, but the span is not being traced, try to start tracing
        if self._exc_info and self._exc_info[0] and not self._is_tracing:
            if self._trace_sampler.lock(ensure=True):
                self._is_tracing = True
                self._signal = self._agent.create_signal()

        # stop profiler, if profiling
        if self._is_profiling:
            try:
                self._operation_profiler.stop(self._signal)
            except Exception as exc:
                logger.error('Error stopping profiler', exc_info=True)
                self._add_profiler_exception(exc)

        # update time and counters
        if not self._is_profiling:
            # only measure time if not profiling due to profiler overhead
            self._metric_store.add_time(duration_us)
        self._metric_store.inc_call_count(1, end_us)
        if self._data is not None:
            for data_name, data in self._data.items():
                data_counts = compute_counts(data)
                for count_name, count in data_counts.items():
                    self._metric_store.inc_data_counter(
                        data_name, count_name, count, end_us)

        # update exception counter
        if self._exc_info and self._exc_info[0]:
            self._metric_store.inc_exception_count(1, end_us)

        # fill and upload profile
        if self._is_tracing:
            if logger.isEnabledFor(logging.DEBUG):
                profiling_stop_overhead_counter = time.perf_counter()

            # read usage data
            try:
                self._agent.read_usage(self._signal)
            except Exception as exc:
                logger.error('Error reading usage information', exc_info=True)
                self._add_profiler_exception(exc)

            # copy data to profile
            self._signal.endpoint = self._endpoint
            self._signal.start_us = end_us - duration_us
            self._signal.end_us = end_us
            if self._exc_info and self._exc_info[0]:
                self._signal.signal_type = signals_pb2.SignalType.EXCEPTION_SIGNAL
            elif self._is_profiling:
                self._signal.signal_type = signals_pb2.SignalType.PROFILE_SIGNAL
            else:
                self._signal.signal_type = signals_pb2.SignalType.SAMPLE_SIGNAL

            # copy tags
            if self._tags is not None:
                for key, value in self._tags.items():
                    tag = self._signal.tags.add()
                    tag.key = key[:50]
                    tag.value = str(value)[:50]

            # copy inference stats
            self._metric_store.finalize(end_us)
            if self._metric_store.latency_us:
                self._signal.span_metrics.latency_us.CopyFrom(
                    self._metric_store.latency_us)
            if self._metric_store.call_count:
                self._signal.span_metrics.call_count.CopyFrom(
                    self._metric_store.call_count)
            if self._metric_store.exception_count:
                self._signal.span_metrics.exception_count.CopyFrom(
                    self._metric_store.exception_count)
            for counter in self._metric_store.data_counters.values():
                self._signal.data_metrics.append(counter)
            self._agent.reset_metric_store(self._endpoint)

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

            # copy data stats
            if self._data is not None:
                for name, data in self._data.items():
                    try:
                        data_stats_proto = build_stats(data)
                        data_stats_proto.data_name = name
                        if data_stats_proto:
                            self._signal.data_stats.append(data_stats_proto)
                    except Exception as exc:
                        logger.error('Error computing data stats', exc_info=True)
                        self._add_profiler_exception(exc)

            # queue signal for upload
            self._agent.uploader().upload_signal(self._signal)
            self._agent.tick()

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('Profiling stop took: %fs', time.perf_counter() - profiling_stop_overhead_counter)

    def set_tag(self, key: str, value: str) -> None:
        if not key:
            raise ValueError('set_tag: key must be provided')
        if value is None:
            raise ValueError('set_tag: value must be provided')

        if self._tags is None:
            self._tags = {}

        if len(self._tags) > TraceSpan.MAX_TAGS:
            raise ValueError('set_tag: too many tags (>{0})'.format(TraceSpan.MAX_TAGS))

        self._tags[key] = value

    def set_exception(self, exc: Optional[Exception] = None, exc_info: Optional[bool] = None) -> None:
        if exc is not None and not isinstance(exc, Exception):
            raise ValueError('set_exception: exc must be instance of Exception')

        if exc_info is not None and not isinstance(exc_info, bool):
            raise ValueError('set_exception: exc_info must be bool')

        if exc:
            self._exc_info = (exc.__class__, str(exc), exc.__traceback__)
        elif exc_info == True:
            self._exc_info = sys.exc_info()

    def set_data(self, name: str, obj: Any) -> None:
        if self._data is None:
            self._data = {}

        if len(self._data) > TraceSpan.MAX_DATA_OBJECTS:
            raise ValueError('set_data: too many data objects (>{0})'.format(TraceSpan.MAX_DATA_OBJECTS))

        if name and not isinstance(name, str):
            raise ValueError('set_data: name must be string')

        self._data[name] = obj

    def _add_profiler_exception(self, exc):
        profiler_error = self._signal.profiler_errors.add()
        profiler_error.message = str(exc)
        if exc.__traceback__:
            frames = traceback.format_tb(exc.__traceback__)
            if len(frames) > 0:
                profiler_error.stack_trace = ''.join(frames)


class Tracer:
    def __init__(self, profiler=None):
        self._profiler = profiler

    def profiler(self):
        return self._profiler

    def span(self,
            endpoint: str,
            tags: Optional[Dict[str, str]] = None,
            ensure_trace: Optional[bool] = False) -> TraceSpan:

        return TraceSpan(
            endpoint=endpoint,
            tags=tags,
            ensure_trace=ensure_trace,
            operation_profiler=self._profiler)

def _timestamp_us():
    return int(time.time() * 1e6)
