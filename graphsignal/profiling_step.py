import logging
import time
from threading import Lock
import traceback

import graphsignal
from graphsignal.proto import profiles_pb2
from graphsignal.profile_scheduler import select_scheduler
from graphsignal.step_counter import get_step_stats, update_step_stats

logger = logging.getLogger('graphsignal')


class ProfilingStep(object):
    __slots__ = [
        '_scheduler',
        '_framework_profiler',
        '_is_scheduled',
        '_is_profiling',
        '_profile',
        '_stop_lock',
        '_run_phase',
        '_effective_batch_size',
        '_start_us'
    ]

    def __init__(self, run_phase=None, effective_batch_size=None, ensure_profile=False, framework_profiler=None):
        self._run_phase = run_phase

        if effective_batch_size is not None and not isinstance(effective_batch_size, int):
            raise Exception('Invalid effective_batch_size')
        self._effective_batch_size = effective_batch_size

        self._scheduler = select_scheduler(run_phase)
        self._framework_profiler = framework_profiler
        self._is_scheduled = False
        self._is_profiling = False
        self._profile = None
        self._stop_lock = Lock()

        if self._scheduler.lock(ensure=ensure_profile):
            self._is_scheduled = True
            self._profile = profiles_pb2.MLProfile()
            self._profile.workload_name = graphsignal._agent.workload_name
            self._profile.worker_id = graphsignal._agent.worker_id
            self._profile.run_id = graphsignal._agent.run_id
            if run_phase:
                self._profile.run_phase = run_phase
            self._profile.node_usage.node_rank = graphsignal._agent.node_rank 
            self._profile.process_usage.local_rank = graphsignal._agent.local_rank 
            self._profile.process_usage.world_rank = graphsignal._agent.world_rank 
            self._profile.process_usage.start_ms = graphsignal._agent.start_ms

            if self._framework_profiler:
                try:
                    self._framework_profiler.start(self._profile)
                    self._is_profiling = True
                except Exception as exc:
                    self._scheduler.unlock()
                    self._is_profiling = False
                    _add_exception(self._profile, exc)
                    logger.error('Error starting profiler', exc_info=True)

            self._profile.start_us = _timestamp_us()

        self._start_us = _timestamp_us()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.stop()

    def stop(self):
        with self._stop_lock:
            step_stats = update_step_stats(
                self._run_phase,
                effective_batch_size=self._effective_batch_size)

            if self._is_scheduled:
                self._profile.end_us = _timestamp_us()

                self._profile.step_stats.step_count = step_stats.step_count
                self._profile.step_stats.sample_count = step_stats.sample_count
                self._profile.step_stats.total_time_us = step_stats.total_time_us

                if graphsignal._agent.params is not None:
                    for name, value in graphsignal._agent.params.items():
                        entry = self._profile.params.add()
                        entry.name = name
                        entry.value = value

                if self._framework_profiler:
                    try:
                        if self._is_profiling:
                            self._framework_profiler.stop(self._profile)
                    except Exception as exc:
                        logger.error('Error stopping profiler', exc_info=True)
                        _add_exception(self._profile, exc)

                try:
                    graphsignal._agent.process_reader.read(self._profile)
                    graphsignal._agent.nvml_reader.read(self._profile)
                except Exception as exc:
                    logger.error('Error reading usage information', exc_info=True)
                    _add_exception(self._profile, exc)

                _upload_profile(self._profile)
                self._is_scheduled = False
                self._is_profiling = False
                self._profile = None
                self._scheduler.unlock()

    def set_effective_batch_size(self, effective_batch_size):
        if not isinstance(effective_batch_size, int):
            raise Exception('Invalid effective_batch_size')
        self._effective_batch_size = effective_batch_size


def _add_exception(profile, exc):
    profiler_error = profile.profiler_errors.add()
    profiler_error.message = str(exc)
    if exc.__traceback__:
        frames = traceback.format_tb(exc.__traceback__)
        if len(frames) > 0:
            profiler_error.stack_trace = ''.join(frames)


def _upload_profile(profile):
    graphsignal._agent.uploader.upload_profile(profile)
    graphsignal._agent.uploader.flush_in_thread()


def _timestamp_us():
    return int(time.time() * 1e6)
