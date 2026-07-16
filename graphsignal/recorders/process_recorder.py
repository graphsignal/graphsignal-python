import logging
import os
import sys
import time

import psutil

import graphsignal
import graphsignal.sdk
from graphsignal.recorders.base_recorder import BaseRecorder

logger = logging.getLogger('graphsignal')


class ProcessRecorder(BaseRecorder):
    def __init__(self, root_pid=None, pid=None, args=None):
        super().__init__(root_pid=root_pid, pid=pid, args=args)
        self._process_start_ts = time.time_ns()
        self._psutil_proc = None
        self._cpu_percent_initialized = False

    def setup(self):
        if self.pid is not None:
            try:
                self._psutil_proc = psutil.Process(self.pid)
                # First call seeds the internal counter; subsequent calls return real values.
                self._psutil_proc.cpu_percent(interval=None)
                self._cpu_percent_initialized = True
                try:
                    create_time = self._psutil_proc.create_time()
                    if create_time:
                        self._process_start_ts = int(create_time * 1e9)
                except psutil.Error:
                    pass
            except psutil.Error:
                self._psutil_proc = None

    def on_tick(self):
        if self._psutil_proc is None:
            return

        sdk = graphsignal.sdk.sdk()
        now_ns = time.time_ns()

        try:
            cpu_percent = self._psutil_proc.cpu_percent(interval=None)
            try:
                cpu_count = psutil.cpu_count() or 1
            except Exception:
                cpu_count = 1
            cpu_percent_normalized = cpu_percent / cpu_count if cpu_count else cpu_percent
            if cpu_percent_normalized > 0:
                sdk.set_gauge(
                    name='process.cpu.usage',
                    value=cpu_percent_normalized,
                    measurement_ts=now_ns,
                    unit='%',
                    tags={'process.pid': str(self.pid)})
        except psutil.Error:
            pass

        try:
            mem = self._psutil_proc.memory_info()
            if mem.rss > 0:
                sdk.set_gauge(
                    name='process.memory.usage',
                    value=mem.rss,
                    measurement_ts=now_ns,
                    tags={'process.pid': str(self.pid)})
            if mem.vms > 0:
                sdk.set_gauge(
                    name='process.memory.virtual',
                    value=mem.vms,
                    measurement_ts=now_ns,
                    tags={'process.pid': str(self.pid)})
        except psutil.Error:
            pass

        process_attrs = {}
        if self.args:
            process_attrs['process.command_line'] = self.args

        is_self = self.pid == os.getpid()
        if is_self:
            try:
                if sys.argv and 'process.command_line' not in process_attrs:
                    process_attrs['process.command_line'] = ' '.join(sys.argv)
            except Exception:
                pass
            try:
                if sys.version_info and len(sys.version_info) >= 3:
                    process_attrs['runtime.name'] = 'python'
                    process_attrs['runtime.version'] = (
                        f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}'
                    )
            except Exception:
                pass

        resource_tags = None
        if self.pid is not None:
            resource_tags = {'process.pid': str(self.pid)}
            # Links every process resource to the watcher's target (root)
            # process — the main server process. The root's own resource has
            # process.pid == process.root_pid, which is how consumers pick
            # the authoritative command line over worker command lines.
            if self.root_pid is not None:
                resource_tags['process.root_pid'] = str(self.root_pid)
        sdk.update_resource(
            'process',
            tags=resource_tags,
            attributes=process_attrs,
            first_seen_ts=self._process_start_ts)
