from typing import Union, Any
import logging
import sys
import time
from threading import Lock
import traceback

import graphsignal
from graphsignal.proto import profiles_pb2

logger = logging.getLogger('graphsignal')


class InferenceSpan:

    __slots__ = [
        '_operation_profiler',
        '_profile_scheduler',
        '_inference_stats',
        '_workload',
        '_model_name',
        '_ensure_profile',
        '_metadata',
        '_context',
        '_is_scheduled',
        '_is_profiling',
        '_profile',
        '_is_stopped',
        '_start_counter',
        '_extra_counts'
    ]

    def __init__(self, 
            model_name,
            metadata=None,
            ensure_profile=False, 
            operation_profiler=None,
            context=None):
        if not model_name:
            raise ValueError('model_name is required')
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError('metadata must be dict')
        self._model_name = model_name
        self._metadata = metadata
        self._ensure_profile = ensure_profile
        self._operation_profiler = operation_profiler
        self._context = context

        self._workload = None
        self._profile_scheduler = None
        self._inference_stats = None
        self._is_stopped = False
        self._is_scheduled = False
        self._is_profiling = False
        self._profile = None
        self._extra_counts = None

        try:
            self._start()
        except Exception as ex:
            if self._is_scheduled:
                self._is_stopped = True
                self._profile_scheduler.unlock()
            raise ex

    def _start(self):
        if self._is_stopped:
            return

        self._workload = graphsignal.workload()
        self._profile_scheduler = self._workload.get_profile_scheduler(self._model_name)
        self._inference_stats = self._workload.get_inference_stats(self._model_name)

        if self._profile_scheduler.lock(ensure=self._ensure_profile):
            if logger.isEnabledFor(logging.DEBUG):
                profiling_start_overhead_counter = time.perf_counter()

            self._is_scheduled = True
            self._profile = self._workload.create_profile()
            self._profile.model_name = self._model_name

            if not graphsignal._agent.disable_op_profiler and self._operation_profiler:
                try:
                    self._operation_profiler.start(self._profile, self._context)
                    self._is_profiling = True
                except Exception as exc:
                    self._add_profiler_exception(exc)
                    logger.error('Error starting profiler', exc_info=True)

            self._profile.start_us = _timestamp_us()

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('Profiling start took: %fs', time.perf_counter() - profiling_start_overhead_counter)

        self._start_counter = time.perf_counter()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.stop(exc_info=exc_info)
        return False

    def stop(self, exc_info=None) -> None:
        try:
            self._stop(exc_info=exc_info)
        finally:
            if self._is_scheduled:
                self._is_stopped = True
                self._profile_scheduler.unlock()            

    def _stop(self, exc_info=None) -> None:
        stop_counter = time.perf_counter()
        end_us = _timestamp_us()

        if self._is_stopped:
            return
        self._is_stopped = True

        if self._is_scheduled:
            self._profile.end_us = end_us

            if self._is_profiling:
                try:
                    self._operation_profiler.stop(self._profile, self._context)
                except Exception as exc:
                    logger.error('Error stopping profiler', exc_info=True)
                    self._add_profiler_exception(exc)

        self._inference_stats.inc_inference_counter(1, end_us)
        if self._extra_counts is not None:
            for name, value in self._extra_counts.items():
                self._inference_stats.inc_extra_counter(name, value, end_us)

        # only measure time if not profiling to exclude profiler overhead
        if not self._is_profiling:
            self._inference_stats.add_time(int((stop_counter - self._start_counter) * 1e6))

        if exc_info and exc_info[0]:
            self._inference_stats.inc_exception_counter(1, end_us)
            self._workload.add_exception(exc_info, end_us)

        if self._is_scheduled:
            if logger.isEnabledFor(logging.DEBUG):
                profiling_stop_overhead_counter = time.perf_counter()

            try:
                graphsignal._agent.process_reader.read(self._profile)
                graphsignal._agent.nvml_reader.read(self._profile)
            except Exception as exc:
                logger.error('Error reading usage information', exc_info=True)
                self._add_profiler_exception(exc)

            self._inference_stats.finalize(end_us)
            for time_us in self._inference_stats.time_reservoir_us:
                self._profile.inference_stats.time_reservoir_us.append(time_us)
            inference_counter_proto = self._profile.inference_stats.inference_counter
            for bucket, count in self._inference_stats.inference_counter.items():
                inference_counter_proto.buckets_sec[bucket] = count
            exception_counter_proto = self._profile.inference_stats.exception_counter
            for bucket, count in self._inference_stats.exception_counter.items():
                exception_counter_proto.buckets_sec[bucket] = count
            for name, counter in self._inference_stats.extra_counters.items():
                extra_counter_proto = self._profile.inference_stats.extra_counters[name]
                for bucket, count in counter.items():
                    extra_counter_proto.buckets_sec[bucket] = count
            self._workload.reset_inference_stats(self._model_name)

            if self._metadata is not None:
                for key, value in self._metadata.items():
                    entry = self._profile.metadata.add()
                    entry.key = key
                    entry.value = str(value)

            for exc_stats in self._workload.exception_stats.values():
                self._profile.exception_stats.append(exc_stats)

            graphsignal._agent.uploader.upload_profile(self._profile)
            self._workload.tick()

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

    def add_metadata(self, key: str, value: Any) -> None:
        if not key:
            raise ValueError('add_metadata: key must be provided')
        if value is None:
            raise ValueError('add_metadata: value must be provided')

        if self._metadata is None:
            self._metadata = {}
        self._metadata[key] = value

    def add_exception(self, exc_info=Union[bool, tuple]) -> None:
        if not exc_info:
            raise ValueError('add_exception: exc_info must be provided')

        if exc_info == True:
            exc_info = sys.exc_info()

        self._workload.add_exception(exc_info, _timestamp_us())

    def _add_profiler_exception(self, exc):
        profiler_error = self._profile.profiler_errors.add()
        profiler_error.message = str(exc)
        if exc.__traceback__:
            frames = traceback.format_tb(exc.__traceback__)
            if len(frames) > 0:
                profiler_error.stack_trace = ''.join(frames)


def _timestamp_us():
    return int(time.time() * 1e6)
