import logging
import time
from threading import Lock
import traceback

import graphsignal
from graphsignal import system_info
from graphsignal.proto import profiles_pb2

logger = logging.getLogger('graphsignal')


class ProfilingSpan(object):
    __slots__ = [
        '_scheduler',
        '_profiler',
        '_is_scheduled',
        '_is_profiling',
        '_profile',
        '_metadata',
        '_stop_lock'
    ]

    SPAN_TIMEOUT_SEC = 10

    def __init__(self, scheduler, profiler, span_name=None, ensure_profile=False):
        self._scheduler = scheduler
        self._profiler = profiler
        self._is_scheduled = False
        self._is_profiling = False
        self._profile = None
        self._metadata = None
        self._stop_lock = Lock()

        if self._scheduler.lock(ensure=ensure_profile):
            self._is_scheduled = True
            self._profile = profiles_pb2.MLProfile()
            self._profile.workload_name = graphsignal._agent.workload_name
            self._profile.run_id = graphsignal._agent.run_id
            self._profile.run_start_ms = graphsignal._agent.run_start_ms
            if span_name:
                self._profile.span_name = span_name

            self._profile.run_env.CopyFrom(system_info.cached_run_env)

            try:
                self._profile.resource_usage.read_us = _timestamp_us()
                graphsignal._agent.host_reader.read(self._profile.resource_usage)
                graphsignal._agent.nvml_reader.read(self._profile.resource_usage)
            except Exception as exc:
                _add_exception(self._profile, exc)
                logger.error(
                    'Error taking run snapshot', exc_info=True)

            try:
                if self._profiler.start():
                    self._is_profiling = True
                else:
                    self._scheduler.unlock()
            except Exception as exc:
                self._scheduler.unlock()
                self._is_profiling = False
                _add_exception(self._profile, exc)
                logger.error('Error starting profiler', exc_info=True)

            self._profile.start_us = _timestamp_us()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.stop()

    def set_name(self, name):
        if self._is_scheduled:
            if name:
                self._profile.span_name = name

    def add_metadata(self, key, value):
        if self._is_scheduled:
            if self._metadata is None:
                self._metadata = {}
            self._metadata[key] = value

    def stop(self, no_save=False):
        with self._stop_lock:
            if self._is_scheduled:
                self._profile.end_us = _timestamp_us()
                if self._metadata is not None:
                    for key, value in self._metadata.items():
                        entry = self._profile.metadata.add()
                        entry.key = str(key)
                        entry.value = str(value)

                try:
                    if self._is_profiling:
                        if self._profiler.stop(self._profile):
                            if not no_save:
                                _upload_profile(self._profile)
                    else:
                        if not no_save:
                            _upload_profile(self._profile)
                except Exception as exc:
                    logger.error('Error stopping profiler', exc_info=True)
                    _add_exception(self._profile, exc)
                    _upload_profile(self._profile)
                finally:
                    self._is_scheduled = False
                    self._is_profiling = False
                    self._profile = None
                    self._scheduler.unlock()


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
