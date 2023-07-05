import logging
import sys
import threading
import time
import math

import graphsignal
from graphsignal import version
from graphsignal.proto import signals_pb2

logger = logging.getLogger('graphsignal')


class LogEntry:
    __slots__ = ('scope', 'name', 'tags', 'level', 'message', 'exception', 'create_ts')

    def __init__(self, scope='user', name=None, tags=None, level='info', message=None, exception=None):
        # no logging in this function!
        self.scope = scope
        self.name = name
        self.tags = tags
        self.level = level
        self.message = message
        self.exception = exception
        self.create_ts = int(time.time())

    def export(self):
        proto = signals_pb2.LogEntry()
        proto.scope = self.scope if self.scope else ''
        proto.name = self.name if self.name else ''
        if self.tags is not None:
            for key, value in self.tags.items():
                tag = proto.tags.add()
                tag.key = str(key)[:50]
                tag.value = str(value)[:250]
        proto.level = self.level if self.level else ''
        proto.message = self.message if self.message else ''
        proto.exception = self.exception if self.exception else ''
        proto.create_ts = self.create_ts

        return proto
    
    def __repr__(self):
        return f'LogEntry(scope={self.scope}, name={self.name}, tags={self.tags}, level={self.level}, message={self.message}, exception={self.exception}, create_ts={self.create_ts})'


class LogStore:
    BUFFER_SIZE = 100
    MESSAGE_SIZE_LIMIT = 1024
    STACK_TRACE_SIZE_LIMIT = 4 * 1024

    def __init__(self):
        self._update_lock = threading.Lock()
        self._has_unexported = False
        self._logs = []

    def log_message(
            self, 
            *,
            scope='user', 
            name=None, 
            tags=None, 
            level='info', 
            message=None, 
            exception=None):
        # no logging in this function!
        if message is None:
            return
        if message and len(message) > self.MESSAGE_SIZE_LIMIT:
            return
        if exception and len(exception) > self.STACK_TRACE_SIZE_LIMIT:
            return
        entry = LogEntry(
            scope=scope, 
            name=name, 
            tags=tags, 
            level=level, 
            message=message, 
            exception=exception)
        with self._update_lock:
            if len(self._logs) > self.BUFFER_SIZE:
                self._logs.pop(0)
            self._logs.append(entry)

    def log_tracer_message(self, 
            tags=None, 
            level=None, 
            message=None, 
            exception=None):
        self.log_message(
            scope='tracer',
            name=f'python-tracer-{version.__version__}',
            tags=tags,
            level=level,
            message=message,
            exception=exception)

    def has_unexported(self):
        return len(self._logs) > 0

    def export(self):
        protos = []
        with self._update_lock:
            for entry in self._logs:
                protos.append(entry.export())
        return protos

    def clear(self):
        with self._update_lock:
            self._logs = []