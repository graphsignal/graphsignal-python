from typing import Dict, Any, Union, Optional
import logging
import sys
import os
import time
import importlib
import socket
import contextvars
import importlib.abc
import threading
import functools
import asyncio
from pathlib import Path

from graphsignal.uploader import Uploader
from graphsignal.metrics import MetricStore
from graphsignal.logs import LogStore
from graphsignal.spans import Span
from graphsignal import client
from graphsignal.utils import uuid_sha1, sanitize_str

logger = logging.getLogger('graphsignal')


class GraphsignalTracerLogHandler(logging.Handler):
    def __init__(self, tracer):
        super().__init__()
        self._tracer = tracer

    def emit(self, record):
        try:
            log_tags = self._tracer.tags.copy()

            exception = None
            if record.exc_info and isinstance(record.exc_info, tuple):
                exception = self.format(record)

            self._tracer.log_store().log_tracer_message(
                tags=log_tags,
                level=record.levelname,
                message=record.getMessage(),
                exception=exception)
        except Exception:
            pass


RECORDER_SPECS = {
    '(default)': [
        ('graphsignal.recorders.process_recorder', 'ProcessRecorder'),
        ('graphsignal.recorders.nvml_recorder', 'NVMLRecorder')],
    'openai': [('graphsignal.recorders.openai_recorder', 'OpenAIRecorder')],
    'torch': [('graphsignal.recorders.pytorch_recorder', 'PyTorchRecorder')]
}

class SourceLoaderWrapper(importlib.abc.SourceLoader):
    def __init__(self, loader, tracer):
        self._loader = loader
        self._tracer = tracer

    def create_module(self, spec):
        return self._loader.create_module(spec)

    def exec_module(self, module):
        self._loader.exec_module(module)
        try:
            self._tracer.initialize_recorders_for_module(module.__name__)
        except Exception:
            logger.error('Error initializing recorders for module %s', module.__name__, exc_info=True)

    def load_module(self, fullname):
        self._loader.load_module(fullname)

    def get_data(self, path):
        return self._loader.get_data(path)

    def get_filename(self, fullname):
        return self._loader.get_filename(fullname)


class SupportedModuleFinder(importlib.abc.MetaPathFinder):
    def __init__(self, tracer):
        self._tracer = tracer
        self._disabled = False

    def find_spec(self, fullname, path=None, target=None):
        if self._disabled:
            return None

        if fullname in RECORDER_SPECS:
            try:
                self._disabled = True
                spec = importlib.util.find_spec(fullname)
                if spec:
                    loader = importlib.util.find_spec(fullname).loader
                    if loader is not None and isinstance(loader, importlib.abc.SourceLoader):
                        return importlib.util.spec_from_loader(fullname, SourceLoaderWrapper(loader, self._tracer))
            except Exception:
                logger.error('Error patching spec for module %s', fullname, exc_info=True)
            finally:
                self._disabled = False

        return None


class Tracer:
    METRIC_READ_INTERVAL_SEC = 10
    METRIC_UPLOAD_INTERVAL_SEC = 20
    LOG_UPLOAD_INTERVAL_SEC = 20
    MAX_PROCESS_TAGS = 25

    def __init__(
            self, 
            api_key=None, 
            api_url=None, 
            tags=None, 
            auto_instrument=True, 
            sampling_rate=1.0,
            profiling_rate=0.1,
            debug_mode=False):
        if debug_mode:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.WARNING)
        
        if not api_key:
            raise ValueError('api_key is required')

        self.api_key = api_key
        if api_url:
            self.api_url = api_url
        else:
            self.api_url = 'https://api.graphsignal.com'
        self.tags = {}
        self.tags.update(tags)
        self.context_tags = contextvars.ContextVar('graphsignal_context_tags', default={})

        self.params = {}

        self.auto_instrument = auto_instrument
        self.sampling_rate = sampling_rate if sampling_rate is not None else 1.0
        self.profiling_rate = profiling_rate if profiling_rate is not None else 0
        self.debug_mode = debug_mode

        self._metric_update_thread = None
        self._tracer_log_handler = None
        self._uploader = None
        self._metric_store = None
        self._log_store = None
        self._recorders = None

        self._process_start_ms = int(time.time() * 1e3)

        self.last_metric_read_ts = 0
        self.last_metric_upload_ts = int(self._process_start_ms / 1e3)
        self.last_log_upload_ts = int(self._process_start_ms / 1e3)

        self.export_on_shutdown = True

    def setup(self):
        self._uploader = Uploader()
        self._uploader.setup()
        self._metric_store = MetricStore()
        self._log_store = LogStore()
        self._recorders = {}

        # initialize tracer log handler
        self._tracer_log_handler = GraphsignalTracerLogHandler(self)
        logger.addHandler(self._tracer_log_handler)

        # initialize module wrappers
        self._module_finder = SupportedModuleFinder(self)
        sys.meta_path.insert(0, self._module_finder)

        # initialize default recorders and recorders for already loaded modules
        for module_name in RECORDER_SPECS.keys():
            if module_name == '(default)' or module_name in sys.modules:
                self.initialize_recorders_for_module(module_name)

    def shutdown(self):
        if self._metric_update_thread:
            self._metric_update_thread.join()
            self._metric_update_thread = None

        if self.export_on_shutdown:
            if self._metric_store.has_unexported():
                metrics = self._metric_store.export()
                for metric in metrics:
                    self._uploader.upload_metric(metric)

            if self._log_store.has_unexported():
                entries = self._log_store.export()
                for entry in entries:
                    self._uploader.upload_log_entry(entry)

        for recorder in self.recorders():
            recorder.shutdown()

        self.upload(block=True)

        self._recorders = None
        self._metric_store = None
        self._log_store = None
        self._uploader = None

        self.tags = None

        self.context_tags.set({})
        self.context_tags = None

        # remove module wrappers
        if self._module_finder in sys.meta_path:
            sys.meta_path.remove(self._module_finder)
        self._module_finder = None

        # remove tracer log handler
        logger.removeHandler(self._tracer_log_handler)
        self._tracer_log_handler = None

    def uploader(self):
        return self._uploader

    def initialize_recorders_for_module(self, module_name):
        # check already loaded
        if module_name in self._recorders:
            return

        # check if supported
        if module_name not in RECORDER_SPECS:
            return

        # load recorder
        logger.debug('Initializing recorder for module: %s', module_name)
        specs = RECORDER_SPECS[module_name]
        self._recorders[module_name] = []
        for spec in specs:
            try:
                recorder_module = importlib.import_module(spec[0])
                recorder_class = getattr(recorder_module, spec[1])
                recorder = recorder_class()
                recorder.setup()
                self._recorders[module_name].append(recorder)
            except Exception:
                logger.error('Failed to initialize recorder for module: %s', module_name, exc_info=True)

    def recorders(self):
        for recorder_list in self._recorders.values():
            for recorder in recorder_list:
                yield recorder

    def metric_store(self):
        return self._metric_store

    def log_store(self):
        return self._log_store

    def emit_span_start(self, span, context):
        last_exc = None
        for recorder in self.recorders():
            try:
                recorder.on_span_start(span, context)
            except Exception as exc:
                last_exc = exc
        if last_exc:
            raise last_exc

    def emit_span_stop(self, span, context):
        last_exc = None
        for recorder in self.recorders():
            try:
                recorder.on_span_stop(span, context)
            except Exception as exc:
                last_exc = exc
        if last_exc:
            raise last_exc

    def emit_span_read(self, span, context):
        last_exc = None
        for recorder in self.recorders():
            try:
                recorder.on_span_read(span, context)
            except Exception as exc:
                last_exc = exc
        if last_exc:
            raise last_exc

    def emit_metric_update(self):
        def on_metric_update():
            last_exc = None
            for recorder in self.recorders():
                try:
                    recorder.on_metric_update()
                except Exception as exc:
                    last_exc = exc
            if last_exc:
                raise last_exc

        self._metric_update_thread = threading.Thread(target=on_metric_update)
        self._metric_update_thread.start()

    def set_tag(self, key: str, value: str, append_uuid: Optional[bool] = False) -> None:
        if not key:
            logger.error('set_tag: key must be provided')
            return

        if value is None:
            self.tags.pop(key, None)
            return

        if len(self.tags) > Tracer.MAX_PROCESS_TAGS:
            logger.error('set_tag: too many tags (>{0})'.format(Tracer.MAX_PROCESS_TAGS))
            return

        if append_uuid:
            if not value:
                value = uuid_sha1(size=12)
            else:
                value = '{0}-{1}'.format(value, uuid_sha1(size=12))

        self.tags[key] = value

    def get_tag(self, key: str) -> Optional[str]:
        return self.tags.get(key, None)

    def remove_tag(self, key: str) -> None:
        self.tags.pop(key, None)

    def set_context_tag(self, key: str, value: str, append_uuid: Optional[bool] = False) -> None:
        if not key:
            logger.error('set_context_tag: key must be provided')
            return

        tags = self.context_tags.get()

        if value is None:
            tags.pop(key, None)
            self.context_tags.set(tags)
            return

        if len(tags) > Tracer.MAX_PROCESS_TAGS:
            logger.error('set_context_tag: too many tags (>{0})'.format(Tracer.MAX_PROCESS_TAGS))
            return

        if append_uuid:
            if not value:
                value = uuid_sha1(size=12)
            else:
                value = '{0}-{1}'.format(value, uuid_sha1(size=12))

        tags[key] = value
        self.context_tags.set(tags)

    def remove_context_tag(self, key: str) -> None:
        tags = self.context_tags.get()
        tags.pop(key, None)
        self.context_tags.set(tags)

    def get_context_tag(self, key: str) -> Optional[str]:
        return self.context_tags.get().get(key, None)

    def set_param(self, name: str, value: Any) -> None:
        if not name:
            logger.error('set_param: name must be provided')
            return

        if value is None:
            self.params.pop(name, None)
            return

        self.params[name] = value

    def get_param(self, name: str) -> Optional[Any]:
        return self.params.get(name, None)

    def remove_param(self, name: str) -> None:
        self.params.pop(name, None)

    def trace(
            self, 
            operation: str,
            tags: Optional[Dict[str, str]] = None,
            with_profile: Optional[bool] = False) -> 'Span':
        return Span(operation=operation, tags=tags, with_profile=with_profile)

    def trace_function(
            self, 
            func=None, 
            *,
            operation: Optional[str] = None,
            tags: Optional[Dict[str, str]] = None,
            with_profile: Optional[bool] = False):
        if func is None:
            return functools.partial(self.trace_function, operation=operation, tags=tags, with_profile=with_profile)

        if operation is None:
            operation_or_name = func.__name__
        else:
            operation_or_name = operation

        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def tf_async_wrapper(*args, **kwargs):
                async with self.trace(operation=operation_or_name, tags=tags, with_profile=with_profile):
                    return await func(*args, **kwargs)
            return tf_async_wrapper
        else:
            @functools.wraps(func)
            def tf_wrapper(*args, **kwargs):
                with self.trace(operation=operation_or_name, tags=tags, with_profile=with_profile):
                    return func(*args, **kwargs)
            return tf_wrapper

    def score(
            self, 
            name: str, 
            tags: Optional[Dict[str, str]] = None,
            score: Optional[Union[int, float]] = None, 
            unit: Optional[str] = None,
            severity: Optional[int] = None,
            comment: Optional[str] = None) -> None:
        now = int(time.time())

        if not name:
            logger.error('score: name is required')
            return

        if not name:
            logger.error('score: score is required')
            return

        model = client.Score(
            score_id=uuid_sha1(size=12),
            tags=[],
            name=name,
            score=score,
            create_ts=now)

        score_tags = {}
        if self.tags is not None:
            score_tags.update(self.tags)
        if self.context_tags:
            score_tags.update(self.context_tags.get().copy())
        if tags is not None:
            score_tags.update(tags)
        for tag_key, tag_value in score_tags.items():
            model.tags.append(client.Tag(
                key=sanitize_str(tag_key, max_len=50),
                value=sanitize_str(tag_value, max_len=250)))

        if unit is not None:
            model.unit = unit

        if severity and severity >= 1 and severity <= 5:
            model.severity = severity

        if comment:
            model.comment = comment

        self.uploader().upload_score(model)
        self.tick(block=False, now=now)

    def upload(self, block=False):
        if block:
            self._uploader.flush()
        else:
            self._uploader.flush_in_thread()

    def check_metric_read_interval(self, now=None):
        if now is None:
            now = time.time()
        return (self.last_metric_read_ts < now - Tracer.METRIC_READ_INTERVAL_SEC)
    
    def set_metric_read(self, now=None):
        self.last_metric_read_ts = now if now else time.time()

    def check_metric_upload_interval(self, now=None):
        if now is None:
            now = time.time()
        return (self.last_metric_upload_ts < now - Tracer.METRIC_UPLOAD_INTERVAL_SEC)

    def set_metric_upload(self, now=None):
        self.last_metric_upload_ts = now if now else time.time()

    def check_log_upload_interval(self, now=None):
        if now is None:
            now = time.time()
        return (self.last_log_upload_ts < now - Tracer.LOG_UPLOAD_INTERVAL_SEC)

    def set_log_upload(self, now=None):
        self.last_log_upload_ts = now if now else time.time()

    def tick(self, block=False, now=None):
        if now is None:
            now = time.time()

        if self.check_metric_upload_interval(now):
            if self._metric_store.has_unexported():
                metrics = self._metric_store.export()
                for metric in metrics:
                    self.uploader().upload_metric(metric)
                self.set_metric_upload(now)

        if self.check_log_upload_interval(now):
            if self._log_store.has_unexported():
                entrys = self._log_store.export()
                for entry in entrys:
                    self.uploader().upload_log_entry(entry)
                self.set_log_upload(now)

        self.upload(block=False)
