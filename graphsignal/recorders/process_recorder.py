
import logging
import os
import sys
import platform
import time
import re
import multiprocessing
import socket
try:
    import resource
except ImportError:
    pass

import graphsignal
from graphsignal.recorders.base_recorder import BaseRecorder

logger = logging.getLogger('graphsignal')

OS_LINUX = (sys.platform.startswith('linux'))
OS_DARWIN = (sys.platform == 'darwin')
OS_WIN = (sys.platform == 'win32')
CPU_NAME_REGEXP = re.compile(r'Model name:\s+(.+)$', flags=re.MULTILINE)
CPU_NAME_MAC_REGEXP = re.compile(r'machdep\.cpu\.brand_string:\s+(.+)$', flags=re.MULTILINE)
VM_RSS_REGEXP = re.compile(r'VmRSS:\s+(\d+)\s+kB')
VM_SIZE_REGEXP = re.compile(r'VmSize:\s+(\d+)\s+kB')
MEM_TOTAL_REGEXP = re.compile(r'MemTotal:\s+(\d+)\s+kB')
MEM_FREE_REGEXP = re.compile(r'MemFree:\s+(\d+)\s+kB')


class ProcessUsage:
    def __init__(self):
        self.pid = 0
        self.rank = 0
        self.has_rank = False
        self.local_rank = 0
        self.has_local_rank = False
        self.start_ms = 0
        self.cpu_name = None
        self.cpu_usage_percent = 0
        self.max_rss = 0
        self.current_rss = 0
        self.vm_size = 0
        self.runtime = None
        self.runtime_version = None
        self.runtime_impl = None


class NodeUsage:
    def __init__(self):
        self.hostname = None
        self.ip_address = None
        self.node_rank = 0
        self.has_node_rank = False
        self.mem_used = 0
        self.mem_total = 0
        self.platform = None
        self.machine = None
        self.os_name = None
        self.os_version = None
        self.num_devices = 0

class SemVer:
    def __init__(self):
        self.major = 0
        self.minor = 0
        self.patch = 0
    
    def __str__(self):
        return f'{self.major}.{self.minor}.{self.patch}'


class ProcessRecorder(BaseRecorder):
    MIN_CPU_READ_INTERVAL_US = 1 * 1e6

    def __init__(self):
        self._last_read_sec = None
        self._last_cpu_time_us = None
        self._last_snapshot = None

    def setup(self):
        self.take_snapshot()

    def on_metric_update(self):
        now = int(time.time())

        process_usage, node_usage = self.take_snapshot()

        store = graphsignal._tracer.metric_store()
        metric_tags = {'deployment': graphsignal._tracer.deployment}
        if graphsignal._tracer.hostname:
            metric_tags['hostname'] = graphsignal._tracer.hostname

        if process_usage.cpu_usage_percent > 0:
            store.set_gauge(
                scope='system', name='process_cpu_usage', tags=metric_tags, 
                value=process_usage.cpu_usage_percent, update_ts=now, unit='%')
        if process_usage.current_rss > 0:
            store.set_gauge(
                scope='system', name='process_memory', tags=metric_tags, 
                value=process_usage.current_rss, update_ts=now, is_size=True)
        if process_usage.vm_size > 0:
            store.set_gauge(
                scope='system', name='virtual_memory', tags=metric_tags, 
                value=process_usage.vm_size, update_ts=now, is_size=True)
        if node_usage.mem_used > 0:
            store.set_gauge(
                scope='system', name='node_memory_used', tags=metric_tags, 
                value=node_usage.mem_used, update_ts=now, is_size=True)

    def on_span_read(self, span, context):
        if self._last_snapshot:
            process_usage, node_usage = self._last_snapshot
            entry = span.config.add()
            entry.key = 'os.name'
            entry.value = node_usage.os_name
            entry = span.config.add()
            entry.key = 'os.version'
            entry.value = node_usage.os_version
            entry = span.config.add()
            entry.key = 'runtime.name'
            entry.value = process_usage.runtime
            entry = span.config.add()
            entry.key = 'runtime.version'
            entry.value = str(process_usage.runtime_version)

    def take_snapshot(self):
        if not OS_WIN:
            rusage_self = resource.getrusage(resource.RUSAGE_SELF)

        if OS_LINUX:
            current_rss = _read_current_rss()
            vm_size = _read_vm_size()

        now = time.time()
        pid = os.getpid()

        node_usage = NodeUsage()
        process_usage = ProcessUsage()

        process_usage.pid = pid

        if not OS_WIN:
            cpu_time_us = _rusage_cpu_time(rusage_self)
            if cpu_time_us is not None:
                if self._last_cpu_time_us is not None:
                    interval_us = (now - self._last_read_sec) * 1e6
                    if interval_us > ProcessRecorder.MIN_CPU_READ_INTERVAL_US:
                        cpu_diff_us = cpu_time_us - self._last_cpu_time_us
                        cpu_usage = (cpu_diff_us / interval_us) * 100
                        try:
                            cpu_usage = cpu_usage / multiprocessing.cpu_count()
                        except Exception:
                            pass
                        process_usage.cpu_usage_percent = cpu_usage

                if (self._last_read_sec is None or 
                        now - self._last_read_sec > ProcessRecorder.MIN_CPU_READ_INTERVAL_US):
                    self._last_read_sec = now
                    self._last_cpu_time_us = cpu_time_us

            if OS_DARWIN:
                max_rss = rusage_self.ru_maxrss
            else:
                max_rss = rusage_self.ru_maxrss * 1e3
            if max_rss is not None:
                process_usage.max_rss = int(max_rss)

        if OS_LINUX:
            if current_rss is not None:
                process_usage.current_rss = current_rss

            if vm_size is not None:
                process_usage.vm_size = vm_size

            mem_total = _read_mem_total()
            if mem_total is not None:
                node_usage.mem_total = mem_total
                mem_free = _read_mem_free()
                if mem_free is not None:
                    node_usage.mem_used = mem_total - mem_free

        try:
            node_usage.hostname = socket.gethostname()
            if node_usage.hostname:
                node_usage.ip_address = socket.gethostbyname(node_usage.hostname)
        except BaseException:
            logger.debug('Error reading hostname', exc_info=True)

        try:
            node_usage.platform = sys.platform
            node_usage.machine = platform.machine()
            if not OS_WIN:
                node_usage.os_name = os.uname().sysname
                node_usage.os_version = os.uname().release
        except BaseException:
            logger.error('Error reading node information', exc_info=True)

        try:
            process_usage.runtime = 'Python'
            process_usage.runtime_version = SemVer()
            process_usage.runtime_version.major = sys.version_info.major
            process_usage.runtime_version.minor = sys.version_info.minor
            process_usage.runtime_version.patch = sys.version_info.micro
            process_usage.runtime_impl = platform.python_implementation()
        except BaseException:
            logger.error('Error reading process information', exc_info=True)

        self._last_snapshot = (process_usage, node_usage)
        return self._last_snapshot


def _rusage_cpu_time(rusage):
    return int((rusage.ru_utime + rusage.ru_stime) * 1e6)  # microseconds


def _read_current_rss():
    pid = os.getpid()

    try:
        f = open('/proc/{0}/status'.format(os.getpid()))
        output = f.read()
        f.close()

        match = VM_RSS_REGEXP.search(output)
        if match:
            return int(float(match.group(1)) * 1e3)
    except Exception:
        pass

    return None


def _read_vm_size():
    pid = os.getpid()

    try:
        f = open('/proc/{0}/status'.format(os.getpid()))
        output = f.read()
        f.close()

        match = VM_SIZE_REGEXP.search(output)
        if match:
            return int(float(match.group(1)) * 1e3)
    except BaseException:
        pass

    return None


def _read_mem_total():
    try:
        f = open('/proc/meminfo')
        output = f.read()
        f.close()

        match = MEM_TOTAL_REGEXP.search(output)
        if match:
            return int(float(match.group(1)) * 1e3)
    except BaseException:
        pass

    return None


def _read_mem_free():
    try:
        f = open('/proc/meminfo')
        output = f.read()
        f.close()

        match = MEM_FREE_REGEXP.search(output)
        if match:
            return int(float(match.group(1)) * 1e3)
    except BaseException:
        pass

    return None
