
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

from graphsignal import client
from graphsignal.recorders.base_recorder import BaseRecorder
import graphsignal
from graphsignal import version

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
        self.pod_uid = None
        self.container_id = None
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
        process_usage, node_usage = self.take_snapshot()

        tracer = graphsignal._tracer
        if platform.system() and platform.release():
            tracer.set_tag('platform.name', platform.system())
            tracer.set_tag('platform.version', platform.release())
        if sys.version_info and len(sys.version_info) >= 3:
            tracer.set_tag('runtime.name', 'python')
            tracer.set_tag('runtime.version', f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')
        if node_usage.hostname:
            tracer.set_tag('host.name', node_usage.hostname)
        if process_usage.pid:
            tracer.set_tag('process.pid', str(process_usage.pid))
        if node_usage.hostname and process_usage.pid:
            tracer.set_tag('process.address', f'{node_usage.hostname}:{process_usage.pid}')
        if node_usage.pod_uid:
            tracer.set_tag('pod.uid', node_usage.pod_uid)
        if node_usage.container_id:
            tracer.set_tag('container.id', node_usage.container_id)

        for env_var in ["RANK", "NCCL_RANK", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK"]:
            if env_var in os.environ:
                tracer.set_tag('process.rank', os.environ[env_var])
                break

        for env_var in ["LOCAL_RANK", "NCCL_LOCAL_RANK", "SLURM_LOCALID", "OMPI_COMM_WORLD_LOCAL_RANK"]:
            if env_var in os.environ:
                tracer.set_tag('process.local_rank', os.environ[env_var])
                break

        tracer.set_tag('tracer.name', f'graphsignal-python')
        tracer.set_tag('tracer.version', version.__version__)

    def on_metric_update(self):
        now = int(time.time())

        process_usage, node_usage = self.take_snapshot()

        store = graphsignal._tracer.metric_store()
        metric_tags = graphsignal._tracer.tags.copy()

        if process_usage.cpu_usage_percent > 0:
            store.set_gauge(
                name='process.cpu.usage', tags=metric_tags, 
                value=process_usage.cpu_usage_percent, update_ts=now, unit='%')
        if process_usage.current_rss > 0:
            store.set_gauge(
                name='process.memory.usage', tags=metric_tags, 
                value=process_usage.current_rss, update_ts=now, is_size=True)
        if process_usage.vm_size > 0:
            store.set_gauge(
                name='process.memory.virtual', tags=metric_tags, 
                value=process_usage.vm_size, update_ts=now, is_size=True)
        if node_usage.mem_used > 0:
            store.set_gauge(
                name='host.memory.usage', tags=metric_tags, 
                value=node_usage.mem_used, update_ts=now, is_size=True)

    def take_snapshot(self):
        if not OS_WIN:
            rusage_self = resource.getrusage(resource.RUSAGE_SELF)

        if OS_LINUX:
            current_rss = _read_current_rss()
            vm_size = _read_vm_size()

        now = time.time()
        node_usage = NodeUsage()
        process_usage = ProcessUsage()

        pid = os.getpid()
        process_usage.pid = pid

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

        pod_uid = os.getenv("POD_UID")
        if pod_uid:
            node_usage.pod_uid = pod_uid

        try:
            container_id = _read_container_id()
            if container_id:
                node_usage.container_id = container_id
        except BaseException:
            logger.error('Error reading container information', exc_info=True)

        try:
            process_usage.runtime = 'Python'
            process_usage.runtime_version = SemVer()
            process_usage.runtime_version.major = sys.version_info.major
            process_usage.runtime_version.minor = sys.version_info.minor
            process_usage.runtime_version.patch = sys.version_info.micro
            process_usage.runtime_impl = platform.python_implementation()
        except BaseException:
            logger.error('Error reading process information', exc_info=True)

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

def _read_container_id():
    container_id = None
    with open( '/proc/self/mountinfo' ) as file:
        line = file.readline().strip()    
        while line:
            if '/docker/containers/' in line:
                container_id = line.split('/docker/containers/')[-1]
                container_id = container_id.split('/')[0]
                break
            line = file.readline().strip()  
    return container_id
