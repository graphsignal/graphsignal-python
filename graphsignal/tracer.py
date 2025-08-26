from typing import Dict, Any, Union, Optional
import logging
import sys
import os
import time
import traceback
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
from graphsignal.utils import fast_rand, uuid_sha1, sanitize_str

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
        ('graphsignal.recorders.python_recorder', 'PythonRecorder'),
        ('graphsignal.recorders.nvml_recorder', 'NVMLRecorder')],
    'torch': [('graphsignal.recorders.pytorch_recorder', 'PyTorchRecorder')],
    'vllm': [('graphsignal.recorders.vllm_recorder', 'VLLMRecorder')]
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

class SamplingTokenBucket:
    def __init__(self, sampling_rate_per_minute: float):
        self.capacity = sampling_rate_per_minute   # max tokens (samples) per minute
        self.tokens = self.capacity               # start full
        self.refill_rate_per_sec = self.capacity / 60.0  # tokens per second
        self.last_refill_time = time.monotonic()
        self._first_request_skipped = False  # Track if first request has been skipped
    
    def _refill(self):
        now = time.monotonic()
        elapsed = now - self.last_refill_time
        if elapsed > 0:
            # Add tokens based on elapsed time
            new_tokens = elapsed * self.refill_rate_per_sec
            self.tokens = min(self.capacity, self.tokens + new_tokens)
            self.last_refill_time = now
    
    def should_sample(self) -> bool:
        # Skip the first request
        if not self._first_request_skipped:
            self._first_request_skipped = True
            return False
        
        # Only refill if we don't have enough tokens
        if self.tokens < 1:
            self._refill()
        
        if self.tokens >= 1:
            self.tokens -= 1
            return True
        return False

class Tracer:
    TICK_INTERVAL_SEC = 10
    MAX_PROCESS_TAGS = 25
    PROFILING_MODE_TIMEOUT_SEC = 60
    MAX_ERRORS_PER_MINUTE = 25
    VALID_ERROR_LEVELS = {'debug', 'info', 'warning', 'error', 'critical'}

    def __init__(
            self, 
            api_key=None, 
            api_url=None, 
            tags=None, 
            auto_instrument=True, 
            samples_per_min=10,
            include_profiles=None,
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
        if tags:
            self.tags.update(tags)
        self.context_tags = contextvars.ContextVar('graphsignal_context_tags', default={})

        self.params = {}

        self.auto_instrument = auto_instrument
        self.samples_per_min = samples_per_min if samples_per_min is not None else 10
        self.include_profiles = include_profiles
        self.debug_mode = debug_mode

        self._sampling_token_buckets = {}

        self._error_counter = 0
        self._error_counter_reset_time = time.time()

        self._tick_timer_thread = None
        self._tick_stop_event = threading.Event()
        self._tick_lock = threading.Lock()
        self._tick_run_thread = None
        self._tracer_log_handler = None
        self._uploader = None
        self._metric_store = None
        self._log_store = None
        self._recorders = None
        self._profiling_mode_lock = threading.Lock()
        self._profiling_mode = None
        self._include_profiles_index = set(include_profiles) if isinstance(include_profiles, list) else None

        self._process_start_ms = int(time.time() * 1e3)

        self.last_tick_ts = time.time()

        self.auto_export = True

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

        # start the tick timer thread
        self._start_tick_timer()

    def _start_tick_timer(self):
        def _tick_loop():
            while not self._tick_stop_event.wait(Tracer.TICK_INTERVAL_SEC):
                try:
                    if self.auto_export:
                        self.tick()
                except Exception as exc:
                    logger.error('Error in tick timer: %s', exc, exc_info=True)

        self._tick_timer_thread = threading.Thread(target=_tick_loop, daemon=True)
        self._tick_timer_thread.start()

    def shutdown(self):
        if self.auto_export:
            self.tick(block=True, force=True)

        if self._tick_stop_event:
            self._tick_stop_event.set()

        if self._tick_timer_thread:
            self._tick_timer_thread.join()
            self._tick_timer_thread = None

        if self._tick_run_thread:
            self._tick_run_thread.join()
            self._tick_run_thread = None

        for recorder in self.recorders():
            recorder.shutdown()

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

    def should_sample(self, sampler_key):
        if sampler_key not in self._sampling_token_buckets:
            self._sampling_token_buckets[sampler_key] = SamplingTokenBucket(self.samples_per_min)
        return self._sampling_token_buckets[sampler_key].should_sample()

    def set_profiling_mode(self, profile_name):
        if not self.should_sample(profile_name):
            return False

        with self._profiling_mode_lock:
            if self._profiling_mode and (time.time() - self._profiling_mode) > self.PROFILING_MODE_TIMEOUT_SEC:
                self._profiling_mode = None

            if self._profiling_mode:
                return False
            else:
                self._profiling_mode = time.time()
                return True

    def unset_profiling_mode(self):
        with self._profiling_mode_lock:
            self._profiling_mode = None

    def is_profiling_mode(self):
        return self._profiling_mode is not None

    def can_include_profiles(self, profiles) -> bool:
        if self._include_profiles_index is None:
            return True
        return any(prof in self._include_profiles_index for prof in profiles)

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
        last_exc = None
        for recorder in self.recorders():
            try:
                recorder.on_metric_update()
            except Exception as exc:
                last_exc = exc
        if last_exc:
            raise last_exc

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
            span_name: str,
            tags: Optional[Dict[str, str]] = None,
            include_profiles: Optional[list] = None) -> 'Span':
        return Span(name=span_name, tags=tags, include_profiles=include_profiles)

    def trace_function(
            self, 
            func=None, 
            *,
            span_name: Optional[str] = None,
            tags: Optional[Dict[str, str]] = None,
            include_profiles: Optional[list] = None):
        if func is None:
            return functools.partial(self.trace_function, span_name=span_name, tags=tags, include_profiles=include_profiles)

        if span_name is None:
            span_or_func_name = func.__name__
        else:
            span_or_func_name = span_name

        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def tf_async_wrapper(*args, **kwargs):
                async with self.trace(span_name=span_or_func_name, tags=tags, include_profiles=include_profiles):
                    return await func(*args, **kwargs)
            return tf_async_wrapper
        else:
            @functools.wraps(func)
            def tf_wrapper(*args, **kwargs):
                with self.trace(span_name=span_or_func_name, tags=tags, include_profiles=include_profiles):
                    return func(*args, **kwargs)
            return tf_wrapper

    def report_error(
            self,
            name: str, 
            tags: Optional[Dict[str, str]] = None,
            level: Optional[str] = None,
            message: Optional[str] = None,
            exc_info: Optional[tuple] = None) -> None:
        now = int(time.time())

        if not name:
            logger.error('error: name is required')
            return

        if level and level not in self.VALID_ERROR_LEVELS:
            logger.error('error: invalid level "%s", must be one of: %s', level, ', '.join(sorted(self.VALID_ERROR_LEVELS)))
            return

        current_time = time.time()
        if current_time - self._error_counter_reset_time >= 60:
            self._error_counter = 0
            self._error_counter_reset_time = current_time
        
        if self._error_counter >= self.MAX_ERRORS_PER_MINUTE:
            logger.warning('Rate limit exceeded: maximum %d errors per minute', self.MAX_ERRORS_PER_MINUTE)
            return
        
        self._error_counter += 1

        model = client.Error(
            error_id=uuid_sha1(size=12),
            tags=[],
            name=name,
            create_ts=now)

        error_tags = {}
        if self.tags is not None:
            error_tags.update(self.tags)
        if self.context_tags:
            error_tags.update(self.context_tags.get().copy())
        if tags is not None:
            error_tags.update(tags)
        for tag_key, tag_value in error_tags.items():
            model.tags.append(client.Tag(
                key=sanitize_str(tag_key, max_len=50),
                value=sanitize_str(tag_value, max_len=250)))

        if level:
            model.level = level

        # Set message from exc_info if provided and no message given
        if exc_info and not message:
            if exc_info[0] and hasattr(exc_info[0], '__name__'):
                message = str(exc_info[0].__name__)
                message += ': '
            if exc_info[1]:
                message += str(exc_info[1])

        if message:
            model.message = message

        # Extract stack trace from exc_info if provided
        if exc_info:
            frames = traceback.format_tb(exc_info[2])
            if len(frames) > 0:
                model.stack_trace = ''.join(frames)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('Uploading error: %s', model)

        self.uploader().upload_error(model)
        self.tick()

    def tick(self, block=False, force=False):
        now = time.time()
        if not force and (now - self.last_tick_ts) < Tracer.TICK_INTERVAL_SEC - 1:
            return
        
        if not self._tick_lock.acquire(blocking=False):
            return

        try:
            def _run_tick():
                try:
                    try:
                        self.emit_metric_update()
                    except Exception as exc:
                        logger.error('Error in metric read event handlers', exc_info=True)

                    if self._metric_store.has_unexported():
                        metrics = self._metric_store.export()
                        for metric in metrics:
                            self.uploader().upload_metric(metric)

                    if self._log_store.has_unexported():
                        entrys = self._log_store.export()
                        for entry in entrys:
                            self.uploader().upload_log_entry(entry)

                    self._uploader.flush()
                except Exception as exc:
                    logger.error('Error in tick execution: %s', exc, exc_info=True)

            self.last_tick_ts = now

            self._tick_run_thread = threading.Thread(target=_run_tick, daemon=True)
            self._tick_run_thread.start()
            if block:
                self._tick_run_thread.join()
        finally:
            self._tick_lock.release()
