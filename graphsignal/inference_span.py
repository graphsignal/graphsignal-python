from typing import Union
import logging
import time
from threading import Lock
import traceback

import graphsignal
from graphsignal.proto import profiles_pb2

logger = logging.getLogger('graphsignal')


class InferenceSpan:
    __slots__ = [
        '_operation_profiler',
        '_context',
        '_is_scheduled',
        '_is_profiling',
        '_profile',
        '_stop_lock',
        '_batch_size',
        '_start_us',
        '_metrics'
    ]

    def __init__(self, 
            batch_size=None, 
            ensure_profile=False, 
            operation_profiler=None,
            context=None):
        if batch_size is not None and not isinstance(batch_size, int):
                raise ValueError('Invalid batch_size')
        self._batch_size = batch_size

        self._operation_profiler = operation_profiler
        self._context = context
        self._is_scheduled = False
        self._is_profiling = False
        self._profile = None
        self._stop_lock = Lock()
        self._metrics = None

        current_run = graphsignal.current_run()

        if current_run.profile_scheduler.lock(ensure=ensure_profile):
            if logger.isEnabledFor(logging.DEBUG):
                profiling_start_ts = time.time()

            self._is_scheduled = True
            self._profile = profiles_pb2.MLProfile()
            self._profile.workload_name = graphsignal._agent.workload_name
            self._profile.worker_id = graphsignal._agent.worker_id
            self._profile.run_id = graphsignal.current_run().run_id
            self._profile.run_start_ms = graphsignal.current_run().start_ms
            self._profile.node_usage.node_rank = graphsignal._agent.node_rank 
            self._profile.process_usage.global_rank = graphsignal._agent.global_rank 
            self._profile.process_usage.local_rank = graphsignal._agent.local_rank 

            try:
                graphsignal._agent.process_reader.start()
                graphsignal._agent.nvml_reader.start()
            except Exception as exc:
                logger.error('Error starting usage readers', exc_info=True)
                self._add_profiler_exception(exc)

            if not graphsignal._agent.disable_op_profiler and self._operation_profiler:
                try:
                    self._operation_profiler.start(self._profile, self._context)
                    self._is_profiling = True
                except Exception as exc:
                    current_run.profile_scheduler.unlock()
                    self._is_profiling = False
                    self._add_profiler_exception(exc)
                    logger.error('Error starting profiler', exc_info=True)

            self._profile.start_us = _timestamp_us()

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('Profiling start took: %fs', time.time() - profiling_start_ts)

        self._start_us = _timestamp_us()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.stop()

    def stop(self) -> None:
        stop_us = _timestamp_us()

        with self._stop_lock:
            if self._is_scheduled:
                if self._is_profiling:
                    try:
                        self._operation_profiler.stop(self._profile, self._context)
                    except Exception as exc:
                        logger.error('Error stopping profiler', exc_info=True)
                        self._add_profiler_exception(exc)

            current_run = graphsignal.current_run()

            current_run.inc_total_inference_count()

            # only measure if not profiling to exclude spans with profiler overhead
            if not self._is_profiling:
                current_run.update_inference_stats(
                    stop_us - self._start_us,
                    batch_size=self._batch_size)

            if self._is_scheduled:
                if logger.isEnabledFor(logging.DEBUG):
                    profiling_stop_ts = time.time()

                self._profile.end_us = stop_us

                self._profile.inference_stats.inference_count = current_run.total_inference_count
                stats = current_run.inference_stats
                self._profile.inference_stats.inference_time_p95_us = stats.inference_time_p95_us()
                self._profile.inference_stats.inference_time_avg_us = stats.inference_time_avg_us()
                self._profile.inference_stats.inference_rate = stats.inference_rate()
                self._profile.inference_stats.sample_rate = stats.sample_rate()
                if self._batch_size:
                    self._profile.inference_stats.batch_size = self._batch_size
                current_run.reset_inference_stats()

                try:
                    graphsignal._agent.process_reader.read(self._profile)
                    graphsignal._agent.nvml_reader.read(self._profile)
                except Exception as exc:
                    logger.error('Error reading usage information', exc_info=True)
                    self._add_profiler_exception(exc)

                current_run.add_profile(self._profile)

                self._is_scheduled = False
                self._is_profiling = False
                self._profile = None
                current_run.profile_scheduler.unlock()

                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug('Profiling stop took: %fs', time.time() - profiling_stop_ts)

    def set_batch_size(self, batch_size: int) -> None:
        if not isinstance(batch_size, int):
            raise ValueError('Invalid batch_size')
        self._batch_size = batch_size

    def _add_profiler_exception(self, exc):
        if self._profile:
            profiler_error = self._profile.profiler_errors.add()
            profiler_error.message = str(exc)
            if exc.__traceback__:
                frames = traceback.format_tb(exc.__traceback__)
                if len(frames) > 0:
                    profiler_error.stack_trace = ''.join(frames)


def _timestamp_us():
    return int(time.time() * 1e6)
