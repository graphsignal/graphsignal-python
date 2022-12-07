
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
from graphsignal.proto import signals_pb2
from graphsignal.proto_utils import parse_semver

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

class ProcessRecorder(BaseRecorder):
    MIN_CPU_READ_INTERVAL_US = 1 * 1e6

    def __init__(self):
        self._last_read_sec = None
        self._last_cpu_time_us = None

    def on_trace_start(self, signal, context):
        if not OS_WIN:
            rusage_thread = resource.getrusage(resource.RUSAGE_THREAD)
            context['start_cpu_time_us'] = _rusage_cpu_time(rusage_thread)

        if OS_LINUX:
            context['start_rss'] = _read_current_rss()

    def on_trace_stop(self, signal, context):
        if not OS_WIN:
            rusage_thread = resource.getrusage(resource.RUSAGE_THREAD)
            stop_cpu_time_us = _rusage_cpu_time(rusage_thread)

            if 'start_cpu_time_us' in context:
                start_cpu_time_us = context['start_cpu_time_us']
                if start_cpu_time_us and stop_cpu_time_us:
                    signal.trace_sample.thread_cpu_time_us = max(0, stop_cpu_time_us - start_cpu_time_us)

        if OS_LINUX:
            current_rss = _read_current_rss()
            if 'start_rss' in context:
                start_rss = context['start_rss']
                if start_rss and current_rss:
                    signal.trace_sample.rss_change = current_rss - start_rss

    def on_trace_read(self, signal, context):
        if not OS_WIN:
            rusage_self = resource.getrusage(resource.RUSAGE_SELF)

        if OS_LINUX:
            current_rss = _read_current_rss()
            vm_size = _read_vm_size()

        now = time.time()
        pid = str(os.getpid())

        node_usage = signal.node_usage
        process_usage = signal.process_usage

        process_usage.process_id = pid

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
            process_usage.runtime = signals_pb2.ProcessUsage.Runtime.PYTHON
            process_usage.runtime_version.major = sys.version_info.major
            process_usage.runtime_version.minor = sys.version_info.minor
            process_usage.runtime_version.patch = sys.version_info.micro
            process_usage.runtime_impl = platform.python_implementation()
        except BaseException:
            logger.error('Error reading process information', exc_info=True)


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
