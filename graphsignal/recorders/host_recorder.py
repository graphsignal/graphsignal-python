import logging
import os
import platform
import socket
import time

import psutil

import graphsignal
import graphsignal.sdk
from graphsignal.recorders.base_recorder import BaseRecorder
from graphsignal.version import __version__ as _profiler_version

logger = logging.getLogger('graphsignal')


class HostRecorder(BaseRecorder):
    def __init__(self, root_pid=None, pid=None, args=None):
        super().__init__(root_pid=root_pid, pid=pid, args=args)
        try:
            self._host_start_ts = int(psutil.boot_time() * 1e9)
        except Exception:
            self._host_start_ts = time.time_ns()

    def setup(self):
        sdk = graphsignal.sdk.sdk()

        try:
            hostname = socket.gethostname()
            if hostname:
                sdk.set_tag('host.name', hostname)
        except Exception:
            logger.debug('Error reading hostname', exc_info=True)

        pod_uid = os.getenv('POD_UID')
        if pod_uid:
            sdk.set_tag('pod.uid', pod_uid)

        container_id = _read_container_id()
        if container_id:
            sdk.set_tag('container.id', container_id)

    def on_tick(self):
        sdk = graphsignal.sdk.sdk()
        now_ns = time.time_ns()

        try:
            vm = psutil.virtual_memory()
            if vm.used > 0:
                sdk.set_gauge(
                    name='host.memory.usage',
                    value=vm.used,
                    measurement_ts=now_ns)
        except Exception:
            pass

        host_attrs = {}
        try:
            if platform.system():
                host_attrs['platform.name'] = platform.system()
            if platform.release():
                host_attrs['platform.version'] = platform.release()
            if platform.machine():
                host_attrs['platform.machine'] = platform.machine()
            # profiler version
            host_attrs['profiler.version'] = _profiler_version
        except Exception:
            pass

        sdk.update_resource(
            'host',
            attributes=host_attrs,
            first_seen_ts=self._host_start_ts)


def _read_container_id():
    try:
        with open('/proc/self/mountinfo') as f:
            for line in f:
                if '/docker/containers/' in line:
                    rest = line.split('/docker/containers/')[-1]
                    return rest.split('/')[0]
    except OSError:
        return None
    return None
