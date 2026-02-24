import logging
import threading
import time

import graphsignal
from graphsignal import version
from graphsignal.proto import signals_pb2

logger = logging.getLogger('graphsignal')


class LogStore:
    BUFFER_SIZE = 100
    MESSAGE_SIZE_LIMIT = 1024
    STACK_TRACE_SIZE_LIMIT = 4 * 1024

    def __init__(self):
        self._log_batches = {}

    def log_message(
            self, 
            *,
            tags=None, 
            level='info', 
            message=None, 
            exception=None,
            timestamp_ns=None):
        # no logging in this function!
        if message is None:
            return
        if message and len(message) > self.MESSAGE_SIZE_LIMIT:
            return
        if exception and len(exception) > self.STACK_TRACE_SIZE_LIMIT:
            return
        
        entry = signals_pb2.LogEntry()
        entry.level = convert_level_from_string(level)
        entry.message = message
        entry.exception = exception if exception else ''
        entry.log_ts = timestamp_ns if timestamp_ns else time.time_ns()

        all_tags = graphsignal._ticker.tags.copy()
        if tags is not None:
            all_tags.update(tags)

        batch_key = frozenset(all_tags.items())
        batch = self._log_batches.get(batch_key)
        if batch is None:
            batch = signals_pb2.LogBatch()

            for key, value in all_tags.items():
                tag = batch.tags.add()
                tag.key = str(key)[:50]
                tag.value = str(value)[:250]
            self._log_batches[batch_key] = batch
        
        batch.log_entries.extend([entry])

    def log_sdk_message(self, 
            tags=None, 
            level=None, 
            message=None, 
            exception=None,
            timestamp_ns=None):

        if tags is None:
            tags = {}
        tags['scope.name'] = 'sdk.python'
        tags['logger'] = 'graphsignal'

        self.log_message(
            tags=tags,
            level=level,
            message=f'Graphsignal {version.__version__}: {message}',
            exception=exception,
            timestamp_ns=timestamp_ns)

    def has_unexported(self):
        return len(self._log_batches) > 0

    def export(self):
        batches = list(self._log_batches.values())
        self._log_batches.clear()
        return batches
 
    def clear(self):
        self._log_batches.clear()

def convert_level_from_string(level: str) -> int:
    if not level:
        return signals_pb2.LogEntry.LogLevel.INFO_LEVEL

    level = level.lower()
    if level == 'debug':
        return signals_pb2.LogEntry.LogLevel.DEBUG_LEVEL
    if level == 'info':
        return signals_pb2.LogEntry.LogLevel.INFO_LEVEL
    if level == 'warning':
        return signals_pb2.LogEntry.LogLevel.WARNING_LEVEL
    if level == 'error':
        return signals_pb2.LogEntry.LogLevel.ERROR_LEVEL
    if level == 'critical':
        return signals_pb2.LogEntry.LogLevel.CRITICAL_LEVEL
    else:
        return signals_pb2.LogEntry.LogLevel.INFO_LEVEL
