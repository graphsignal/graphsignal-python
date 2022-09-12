from typing import Union, Any, Optional, Dict
import logging
import sys
import time
import traceback

import graphsignal
from graphsignal.proto import signals_pb2
from graphsignal.data import compute_size, compute_stats 

logger = logging.getLogger('graphsignal')


class InferenceSpan:
    MAX_TAGS = 10
    MAX_DATA_OBJECTS = 10

    __slots__ = [
        '_operation_profiler',
        '_trace_sampler',
        '_span_stats',
        '_agent',
        '_model_name',
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
            model_name,
            tags=None,
            ensure_trace=False, 
            operation_profiler=None):
        if not model_name:
            raise ValueError('model_name is required')
        if not isinstance(model_name, str):
            raise ValueError('model_name must be string')
        if len(model_name) > 50:
            raise ValueError('model_name is too long (>50)')
        if tags is not None:
            if not isinstance(tags, dict):
                raise ValueError('tags must be dict')
            if len(tags) > InferenceSpan.MAX_TAGS:
                raise ValueError('too many tags (>{0})'.format(InferenceSpan.MAX_TAGS))

        self._model_name = model_name
        self._tags = tags
        self._ensure_trace = ensure_trace
        self._operation_profiler = operation_profiler

        self._agent = None
        self._trace_sampler = None
        self._span_stats = None
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
        self._trace_sampler = self._agent.get_trace_sampler(self._model_name)
        self._span_stats = self._agent.get_span_stats(self._model_name)

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
            # only measure time if not profiling to exclude profiler overhead
            self._span_stats.add_time(duration_us)
        self._span_stats.inc_call_counter(1, end_us)
        if self._data is not None:
            for name, data in self._data.items():
                data_size, size_unit = compute_size(data)
                self._span_stats.inc_data_counter(name, data_size, size_unit, end_us)

        # update exception counter
        if self._exc_info and self._exc_info[0]:
            self._span_stats.inc_exception_counter(1, end_us)

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
            self._signal.model_name = self._model_name
            self._signal.start_us = end_us - duration_us
            self._signal.end_us = end_us
            if self._exc_info and self._exc_info[0]:
                self._signal.signal_type = signals_pb2.SignalType.INFERENCE_EXCEPTION_SIGNAL
            elif self._is_profiling:
                self._signal.signal_type = signals_pb2.SignalType.INFERENCE_PROFILE_SIGNAL
            else:
                self._signal.signal_type = signals_pb2.SignalType.INFERENCE_SAMPLE_SIGNAL

            # copy tags
            if self._tags is not None:
                for key, value in self._tags.items():
                    tag = self._signal.tags.add()
                    tag.key = key[:50]
                    tag.value = str(value)[:50]

            # copy inference stats
            self._span_stats.finalize(end_us)
            self._signal.span_stats.time_reservoir_us[:] = \
                self._span_stats.time_reservoir_us
            self._signal.span_stats.call_counter.CopyFrom(
                self._span_stats.call_counter)
            self._signal.span_stats.exception_counter.CopyFrom(
                self._span_stats.exception_counter)
            for name, counter in self._span_stats.data_counters.items():
                self._signal.span_stats.data_counters[name].CopyFrom(counter)
            self._agent.reset_span_stats(self._model_name)

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
                        data_stats_proto = compute_stats(data)
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

        if len(self._tags) > InferenceSpan.MAX_TAGS:
            raise ValueError('set_tag: too many tags (>{0})'.format(InferenceSpan.MAX_TAGS))

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

    def set_data(self, name: str, data: Any) -> None:
        if self._data is None:
            self._data = {}

        if len(self._data) > InferenceSpan.MAX_DATA_OBJECTS:
            raise ValueError('set_data: too many data objects (>{0})'.format(InferenceSpan.MAX_DATA_OBJECTS))

        if name and not isinstance(name, str):
            raise ValueError('set_data: name must be string')

        self._data[name] = data

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

    def inference_span(self,
        model_name: str,
        tags: Optional[dict] = None,
        ensure_trace: Optional[bool] = False) -> InferenceSpan:

        return InferenceSpan(
            model_name=model_name,
            tags=tags,
            ensure_trace=ensure_trace,
            operation_profiler=self._profiler)


def _timestamp_us():
    return int(time.time() * 1e6)
