from typing import Union, Any, Optional
import logging
import sys
import time
import traceback

import graphsignal
from graphsignal.proto import signals_pb2

logger = logging.getLogger('graphsignal')


class InferenceSpan:
    MAX_TAGS = 10

    __slots__ = [
        '_operation_profiler',
        '_trace_sampler',
        '_inference_stats',
        '_agent',
        '_model_name',
        '_ensure_trace',
        '_tags',
        '_is_tracing',
        '_is_profiling',
        '_signal',
        '_is_stopped',
        '_start_counter',
        '_extra_counts',
        '_exc_info'
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
        self._inference_stats = None
        self._is_stopped = False
        self._is_tracing = False
        self._is_profiling = False
        self._signal = None
        self._extra_counts = None
        self._exc_info = None

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
        self._inference_stats = self._agent.get_inference_stats(self._model_name)

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
            self._inference_stats.add_time(duration_us)
        self._inference_stats.inc_inference_counter(1, end_us)
        if self._extra_counts is not None:
            for name, value in self._extra_counts.items():
                self._inference_stats.inc_extra_counter(name, value, end_us)

        # update exception counter
        if self._exc_info and self._exc_info[0]:
            self._inference_stats.inc_exception_counter(1, end_us)

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

            # copy inference stats
            self._inference_stats.finalize(end_us)
            for time_us in self._inference_stats.time_reservoir_us:
                self._signal.inference_stats.time_reservoir_us.append(time_us)
            inference_counter_proto = self._signal.inference_stats.inference_counter
            for bucket, count in self._inference_stats.inference_counter.items():
                inference_counter_proto.buckets_sec[bucket] = count
            exception_counter_proto = self._signal.inference_stats.exception_counter
            for bucket, count in self._inference_stats.exception_counter.items():
                exception_counter_proto.buckets_sec[bucket] = count
            for name, counter in self._inference_stats.extra_counters.items():
                extra_counter_proto = self._signal.inference_stats.extra_counters[name]
                for bucket, count in counter.items():
                    extra_counter_proto.buckets_sec[bucket] = count
            self._agent.reset_inference_stats(self._model_name)

            # copy tags
            if self._tags is not None:
                for key, value in self._tags.items():
                    tag = self._signal.tags.add()
                    tag.key = key[:50]
                    tag.value = str(value)[:50]

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

            # queue signal for upload
            self._agent.uploader().upload_signal(self._signal)
            self._agent.tick()

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('Profiling stop took: %fs', time.perf_counter() - profiling_stop_overhead_counter)

    def set_count(self, name: str, value: Union[int, float]) -> None:
        if not name:
            raise ValueError('set_count: name must be provided')
        if not isinstance(value, (int, float)):
            raise ValueError('set_count: value must be int or float')

        if self._extra_counts is None:
            self._extra_counts = {}
        self._extra_counts[name] = value

    def add_tag(self, key: str, value: Any) -> None:
        if not key:
            raise ValueError('add_tag: key must be provided')
        if value is None:
            raise ValueError('add_tag: value must be provided')

        if self._tags is not None and len(self._tags) > InferenceSpan.MAX_TAGS:
            return

        if self._tags is None:
            self._tags = {}
        self._tags[key] = value

    def set_exception(self, exc_info=Union[bool, tuple]) -> None:
        if not exc_info:
            raise ValueError('set_exception: exc_info must be provided')

        if exc_info == True:
            exc_info = sys.exc_info()

        self._exc_info = exc_info

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
