
import logging
import os
import sys
import time
import resource
import re
import multiprocessing

import graphsignal
from graphsignal.proto import profiles_pb2

logger = logging.getLogger('graphsignal')

OS_LINUX = (sys.platform.startswith('linux'))
OS_DARWIN = (sys.platform == 'darwin')
OS_WIN = (sys.platform == 'win32')
VM_RSS_REGEXP = re.compile('VmRSS:\\s+(\\d+)\\s+kB')
VM_SIZE_REGEXP = re.compile('VmSize:\\s+(\\d+)\\s+kB')


class HostReader():
    __slots__ = [
        '_last_read_time',
        '_last_cpu_time_ns'
    ]

    MIN_CPU_READ_INTERVAL = 1e9

    def __init__(self):
        self._last_read_time = None
        self._last_cpu_time_ns = None

    def setup(self):
        pass

    def shutdown(self):
        pass

    def read(self, resource_usage):
        host_usage = resource_usage.host_usage

        if not OS_WIN:
            cpu_time_ns = _read_cpu_time()
            if cpu_time_ns is not None:
                if self._last_cpu_time_ns is not None:
                    interval_ns = (time.time() - self._last_read_time) * 1e9
                    if interval_ns > HostReader.MIN_CPU_READ_INTERVAL:
                        cpu_diff_ns = cpu_time_ns - self._last_cpu_time_ns
                        cpu_usage = (cpu_diff_ns / interval_ns) * 100
                        try:
                            cpu_usage = cpu_usage / multiprocessing.cpu_count()
                        except Exception:
                            pass
                        host_usage.cpu_usage_percent = cpu_usage
                else:
                    self._last_read_time = time.time()
                    self._last_cpu_time_ns = cpu_time_ns

        if not OS_WIN:
            max_rss = _read_max_rss()
            if max_rss is not None:
                host_usage.max_rss = max_rss

        if OS_LINUX:
            current_rss = _read_current_rss()
            if current_rss is not None:
                host_usage.current_rss = current_rss

            vm_size = _read_vm_size()
            if vm_size is not None:
                host_usage.vm_size = vm_size


def _read_cpu_time():
    rusage = resource.getrusage(resource.RUSAGE_SELF)
    return int((rusage.ru_utime + rusage.ru_stime) * 1e9)  # ns


def _read_max_rss():
    rusage = resource.getrusage(resource.RUSAGE_SELF)

    if OS_DARWIN:
        return rusage.ru_maxrss
    else:
        return rusage.ru_maxrss * 1024


def _read_current_rss():
    pid = os.getpid()

    output = None
    try:
        f = open('/proc/{0}/status'.format(os.getpid()))
        output = f.read()
        f.close()
    except Exception:
        return None

    match = VM_RSS_REGEXP.search(output)
    if match:
        return int(float(match.group(1)) * 1e3)

    return None


def _read_vm_size():
    pid = os.getpid()

    output = None
    try:
        f = open('/proc/{0}/status'.format(os.getpid()))
        output = f.read()
        f.close()
    except Exception:
        return None

    match = VM_SIZE_REGEXP.search(output)
    if match:
        return int(float(match.group(1)) * 1e3)

    return None
