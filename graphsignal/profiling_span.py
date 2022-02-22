import logging
import time
from threading import Lock
import traceback

import graphsignal
from graphsignal import system_info
from graphsignal import profiles_pb2

logger = logging.getLogger('graphsignal')


class ProfilingSpan(object):
    __slots__ = [
        '_scheduler',
        '_profiler',
        '_span_name',
        '_is_profiling',
        '_metadata',
        '_start_us',
        '_start_exc',
        '_stop_lock'
    ]

    SPAN_TIMEOUT_SEC = 10

    def __init__(self, scheduler, profiler, span_name=None,
                 ensure_profile=False):
        self._scheduler = scheduler
        self._profiler = profiler
        self._span_name = span_name
        self._is_profiling = False
        self._metadata = None
        self._start_us = None
        self._start_exc = None
        self._stop_lock = Lock()

        if self._scheduler.lock(ensure=ensure_profile):
            try:
                if self._profiler.start():
                    self._start_us = _timestamp_us()
                    self._is_profiling = True
                else:
                    self._scheduler.unlock()
            except Exception as exc:
                self._scheduler.unlock()
                self._is_profiling = False
                self._start_us = _timestamp_us()
                self._start_exc = exc
                logger.error('Error starting profiler', exc_info=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.stop()

    def set_name(self, name):
        self._span_name = name

    def add_metadata(self, key, value):
        if self._metadata is None:
            self._metadata = {}
        self._metadata[key] = value

    def _check_stop(self):
        if self._is_profiling:
            self.stop()

    def stop(self, no_save=False):
        with self._stop_lock:
            if self._is_profiling or self._start_exc:
                end_us = _timestamp_us()

                try:
                    profile = profiles_pb2.MLProfile()
                    profile.workload_name = graphsignal._agent.workload_name
                    profile.run_id = graphsignal._agent.run_id
                    profile.span_name = self._span_name if self._span_name else ''
                    profile.start_us = self._start_us
                    profile.end_us = end_us

                    profile.run_env.CopyFrom(system_info.cached_run_env)

                    if self._metadata is not None:
                        for key, value in self._metadata.items():
                            entry = profile.metadata.add()
                            entry.key = str(key)
                            entry.value = str(value)

                    if self._is_profiling:
                        if self._profiler.stop(profile):
                            if not no_save:
                                _upload_profile(profile)
                    elif self._start_exc:
                        _add_exception(profile, self._start_exc)
                        if not no_save:
                            _upload_profile(profile)
                except Exception as exc:
                    logger.error('Error stopping profiler', exc_info=True)
                    _add_exception(profile, exc)
                    _upload_profile(profile)
                finally:
                    self._is_profiling = False
                    self._start_exc = None
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
