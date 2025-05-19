import logging
import os
import sys
import cProfile, pstats
import threading
import json

import graphsignal
from graphsignal.recorders.base_recorder import BaseRecorder

logger = logging.getLogger('graphsignal')

class PythonRecorder(BaseRecorder):
    def __init__(self):
        self._profiler = None
        self._exclude_path = os.path.dirname(os.path.realpath(graphsignal.__file__))

    def _can_include_profiles(self, span, profiles):
        return (graphsignal._tracer.can_include_profiles(profiles) and 
                span.can_include_profiles(profiles))

    def on_span_start(self, span, context):
        if (span.sampled() and 
            self._can_include_profiles(span, ['python-time-profile']) and 
            graphsignal._tracer.set_profiling_mode()):

            context['profiled'] = True

            if self._profiler:
                # In case of previous profiling not stopped
                self._profiler.disable()
                self._profiler = None
            self._profiler = cProfile.Profile()
            self._profiler.enable()

    def on_span_stop(self, span, context):
        if context.get('profiled', False):
            graphsignal._tracer.unset_profiling_mode()
            if self._profiler:
                self._profiler.disable()

    def on_span_read(self, span, context):
        if context.get('profiled', False):
            if self._profiler:
                try:
                    self._convert_to_profile(span)
                finally:
                    self._profiler = None

    def _convert_to_profile(self, span):
        stats = pstats.Stats(self._profiler)

        stats.sort_stats('tottime')
        func_keys = stats.fcn_list[:] if stats.fcn_list else list(stats.stats.keys())

        func_list = []
        visited = {}
        for func_key in func_keys:
            func_stats = stats.stats[func_key]
            file_name, line_num, func_name = func_key
            if self._has_exclude_func(stats.stats, func_key, visited):
                continue
            cc, nc, tt, ct, _ = func_stats
            func_list.append((file_name, line_num, func_name, nc, tt, ct))

        wt_profile = []
        for file_name, line_num, func_name, nc, tt, ct in func_list[:250]:
            wt_profile.append(dict(
                func_name = _format_frame(file_name, line_num, func_name),
                count = int(nc),
                wall_time_ns = _ns(ct),
                self_wall_time_ns = _ns(tt),
            ))

        if len(wt_profile) > 0:
            span.set_profile(
                name='python-time-profile', 
                format='event-averages', 
                content=json.dumps(wt_profile))

        if len(wt_profile) > 0:
            if sys.version_info:
                span.set_param('profiler', f'cprofile-{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')

    def _has_exclude_func(self, stats, func_key, visited):
        if func_key in visited:
            return visited[func_key]

        file_name, _, _ = func_key
        _, _, _, _, callers = stats[func_key]

        if file_name.startswith(self._exclude_path):
            visited[func_key] = True
            return True
        else:
            visited[func_key] = False

        for caller_key in callers.keys():
            if self._has_exclude_func(stats, caller_key, visited):
                visited[func_key] = True
                return True

        return False


def _ns(sec):
    return int(sec * 1e9)


def _format_frame(file_name, line_num, func_name):
    if file_name == '~':
        file_name = ''

    if file_name and line_num and func_name:
        return '{func_name} ({file_name}:{line_num})'.format(
            file_name=file_name,
            func_name=func_name,
            line_num=line_num)
    elif file_name and func_name:
        return '{func_name} ({file_name})'.format(
            file_name=file_name,
            func_name=func_name)
    elif func_name:
        return func_name
    else:
        return 'unknown'