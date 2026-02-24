from token import OP
from typing import Dict, Optional
import logging
import sys
import os
import time
import importlib
import contextvars
import importlib.abc
import threading
import functools
import asyncio

from graphsignal.core.signal_uploader import SignalUploader
from graphsignal.core.config_loader import ConfigLoader
from graphsignal.signals.metrics import MetricStore
from graphsignal.signals.logs import LogStore
from graphsignal.signals.spans import Span
from graphsignal.core.sampler import TimeCoordinatedSampler
from graphsignal.profilers.function_profiler import FunctionProfiler
from graphsignal.profilers.cupti_profiler import CuptiProfiler
from graphsignal.proto import signals_pb2
from graphsignal.utils import uuid_sha1

logger = logging.getLogger('graphsignal')



class GraphsignalLogHandler(logging.Handler):
    def __init__(self, ticker):
        super().__init__()
        self._ticker = ticker

    def emit(self, record):
        try:
            log_tags = self._ticker.tags.copy()

            exception = None
            if record.exc_info and isinstance(record.exc_info, tuple):
                exception = self.format(record)

            self._ticker.log_store().log_sdk_message(
                tags=log_tags,
                level=record.levelname,
                message=record.getMessage(),
                exception=exception)
        except Exception:
            pass


RECORDER_SPECS = {
    '(default)': [
        ('graphsignal.recorders.process_recorder', 'ProcessRecorder'),
        ('graphsignal.recorders.exception_recorder', 'ExceptionRecorder'),
        ('graphsignal.recorders.nvml_recorder', 'NVMLRecorder')],
    'torch': [('graphsignal.recorders.pytorch_recorder', 'PyTorchRecorder')],
    'vllm': [('graphsignal.recorders.vllm_recorder', 'VLLMRecorder')]
}

class SourceLoaderWrapper(importlib.abc.SourceLoader):
    def __init__(self, loader, ticker):
        self._loader = loader
        self._ticker = ticker

    def create_module(self, spec):
        return self._loader.create_module(spec)

    def exec_module(self, module):
        self._loader.exec_module(module)
        try:
            self._ticker.initialize_recorders_for_module(module.__name__)
        except Exception:
            logger.error('Error initializing recorders for module %s', module.__name__, exc_info=True)

    def load_module(self, fullname):
        self._loader.load_module(fullname)

    def get_data(self, path):
        return self._loader.get_data(path)

    def get_filename(self, fullname):
        return self._loader.get_filename(fullname)


class SupportedModuleFinder(importlib.abc.MetaPathFinder):
    def __init__(self, ticker):
        self._ticker = ticker
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
                        return importlib.util.spec_from_loader(fullname, SourceLoaderWrapper(loader, self._ticker))
            except Exception:
                logger.error('Error patching spec for module %s', fullname, exc_info=True)
            finally:
                self._disabled = False

        return None

class Ticker:
    TICK_DELAY_SEC = 2
    TICK_INTERVAL_SEC = 10
    MAX_PROCESS_TAGS = 25
    MAX_SAMPLERS = 100

    def __init__(
            self, 
            api_key=None, 
            api_url=None, 
            tags=None, 
            auto_instrument=True, 
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

        self.auto_instrument = auto_instrument
        self.debug_mode = debug_mode
        if debug_mode:
            self._debug_mode_forced = True
        else:
            self._debug_mode_forced = False

        self._samplers = {}

        self._tick_timer_thread = None
        self._tick_stop_event = threading.Event()
        self._tick_lock = threading.Lock()
        self._tick_run_thread = None
        self._python_log_handler = None
        self._config_loader = None
        self._signal_uploader = None
        self._metric_store = None
        self._log_store = None
        self._function_profiler = None
        self._cupti_profiler = None
        self._recorders = None

        self._process_start_ms = int(time.time() * 1e3)

        self.last_tick_ts = time.time()

        self.auto_tick = True

    def setup(self):
        logger.debug('SDK setup started')

        # initialize config loader
        self._config_loader = ConfigLoader()
        self._config_loader.setup()
        def update_func(changed_options):
            if 'traces_per_sec' in changed_options:
                # new samplers will pick up the new rate
                self._samplers.clear()
            if 'debug_mode' in changed_options:
                if not self._debug_mode_forced:
                    self._set_debug_mode_from_config()
        self._config_loader.on_update(update_func)

        # initialize signal uploader
        self._signal_uploader = SignalUploader()
        self._signal_uploader.setup()

        # initialize metric store
        self._metric_store = MetricStore()

        # initialize log store
        self._log_store = LogStore()

        # initialize ticker log handler early so that all SDK components
        # (including profilers) can emit logs into the log store.
        if not self._python_log_handler:
            self._python_log_handler = GraphsignalLogHandler(self)
            logger.addHandler(self._python_log_handler)

        # initialize function profiler
        self._function_profiler = FunctionProfiler(profile_name='profile.python')
        self._function_profiler.setup()

        # initialize CUPTI activity profiler
        self._cupti_profiler = CuptiProfiler(profile_name='profile.cuda', debug_mode=self.debug_mode)
        self._cupti_profiler.setup()

        # initialize module wrappers
        self._module_finder = SupportedModuleFinder(self)
        sys.meta_path.insert(0, self._module_finder)

        # initialize default recorders and recorders for already loaded modules
        self._recorders = {}
        for module_name in RECORDER_SPECS.keys():
            if module_name == '(default)' or module_name in sys.modules:
                self.initialize_recorders_for_module(module_name)

        # start the tick timer thread
        self._start_tick_timer()

        # register fork handler to reinitialize in child processes
        if hasattr(os, 'register_at_fork'):
            os.register_at_fork(after_in_child=self._reinitialize_after_fork)

        logger.debug('SDK setup complete')

    def _start_tick_timer(self):
        self._tick_stop_event = threading.Event()

        def _tick_loop():
            if not self._tick_stop_event.wait(Ticker.TICK_DELAY_SEC):
                try:
                    if self.auto_tick:
                        self.tick(force=True)
                except Exception as exc:
                    logger.error('Error in initial tick: %s', exc, exc_info=True)
            
            while not self._tick_stop_event.wait(Ticker.TICK_INTERVAL_SEC):
                try:
                    if self.auto_tick:
                        self.tick()
                except Exception as exc:
                    logger.error('Error in tick timer: %s', exc, exc_info=True)

        self._tick_timer_thread = threading.Thread(target=_tick_loop, daemon=True)
        self._tick_timer_thread.start()

    def _stop_tick_timer(self):
        if self._tick_timer_thread:
            self._tick_stop_event.set()
            self._tick_timer_thread.join()
            self._tick_stop_event = None
            self._tick_timer_thread = None

    def _reinitialize_after_fork(self):
        logger.debug('SDK reinitialization after fork started')

        try:
            self._stop_tick_timer()
            
            if self._function_profiler:
                try:
                    self._function_profiler._stop_rollover_timer()
                    self._function_profiler._start_rollover_timer()
                except Exception:
                    pass

            if self._cupti_profiler:
                try:
                    # CUPTI state does not survive fork safely; restart in the child.
                    self._cupti_profiler.shutdown()
                    self._cupti_profiler = CuptiProfiler(profile_name='profile.cuda', debug_mode=self.debug_mode)
                    self._cupti_profiler.setup()
                    if self._cupti_profiler._disabled:
                        self._cupti_profiler = None
                except Exception:
                    pass
            
            if self._metric_store:
                try:
                    self._metric_store.clear()
                except Exception:
                    pass
            
            if self._log_store:
                try:
                    self._log_store.clear()
                except Exception:
                    pass
            
            if self._signal_uploader:
                try:
                    self._signal_uploader._buffer_lock = threading.Lock()
                    self._signal_uploader._flush_lock = threading.Lock()
                    self._signal_uploader.clear()
                except Exception:
                    pass

            self._start_tick_timer()
            
            logger.debug('SDK reinitialized after fork')
        except Exception as exc:
            logger.error('Error reinitializing SDK after fork: %s', exc, exc_info=True)
        
        logger.debug('SDK reinitialization after fork complete')

    def shutdown(self):
        if self.auto_tick:
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

        if self._function_profiler:
            self._function_profiler.shutdown()
            self._function_profiler = None

        if self._cupti_profiler:
            self._cupti_profiler.shutdown()
            self._cupti_profiler = None

        self._recorders = None
        self._metric_store = None
        self._log_store = None
        self._signal_uploader = None

        if self._config_loader:
            self._config_loader.shutdown()
            self._config_loader = None

        self.tags = None

        self.context_tags.set({})
        self.context_tags = None

        # remove module wrappers
        if self._module_finder in sys.meta_path:
            sys.meta_path.remove(self._module_finder)
        self._module_finder = None

        # remove ticker log handler
        logger.removeHandler(self._python_log_handler)
        self._python_log_handler = None

    def config_loader(self):
        return self._config_loader

    def _set_debug_mode_from_config(self) -> None:
        try:
            debug_mode_int = self.config_loader().get_int_option('debug_mode')
            if debug_mode_int is None:
                return
            debug_mode = debug_mode_int == 1

            self.debug_mode = debug_mode
            logger.setLevel(logging.DEBUG if debug_mode else logging.WARNING)

            if self._cupti_profiler:
                try:
                    self._cupti_profiler.set_debug_mode(debug_mode)
                except Exception:
                    pass
        except Exception as exc:
            logger.error('Error applying debug mode from config: %s', exc, exc_info=True)

    def signal_uploader(self):
        return self._signal_uploader

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

    def function_profiler(self):
        return self._function_profiler

    def metric_store(self):
        return self._metric_store

    def log_store(self):
        return self._log_store

    def sampler(self, sampler_key):
        if sampler_key not in self._samplers:
            if len(self._samplers) >= self.MAX_SAMPLERS:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug('Maximum number of samplers reached, skipping sampler: %s', sampler_key)
                return None
            
            traces_per_sec = self.config_loader().get_float_option('traces_per_sec')
            if traces_per_sec is None or traces_per_sec == 0:
                return None
            
            self._samplers[sampler_key] = TimeCoordinatedSampler(traces_per_sec)

        return self._samplers[sampler_key]

    def should_trace(self, sampler_key):
        sampler = self.sampler(sampler_key)
        if not sampler:
            return False
        return sampler.should_sample()

    def emit_tick(self):
        last_exc = None
        for recorder in self.recorders():
            try:
                recorder.on_tick()
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

        if len(self.tags) > Ticker.MAX_PROCESS_TAGS:
            logger.error('set_tag: too many tags (>{0})'.format(Ticker.MAX_PROCESS_TAGS))
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

        if len(tags) > Ticker.MAX_PROCESS_TAGS:
            logger.error('set_context_tag: too many tags (>{0})'.format(Ticker.MAX_PROCESS_TAGS))
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

    def trace(
            self, 
            span_name: str,
            tags: Optional[Dict[str, str]] = None) -> 'Span':
        return Span(name=span_name, tags=tags)

    def trace_function(
            self, 
            func=None, 
            *,
            span_name: Optional[str] = None,
            tags: Optional[Dict[str, str]] = None):
        if func is None:
            return functools.partial(self.trace_function, span_name=span_name, tags=tags)

        if span_name is None:
            span_or_func_name = func.__name__
        else:
            span_or_func_name = span_name

        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def tf_async_wrapper(*args, **kwargs):
                async with self.trace(span_name=span_or_func_name, tags=tags):
                    return await func(*args, **kwargs)
            return tf_async_wrapper
        else:
            @functools.wraps(func)
            def tf_wrapper(*args, **kwargs):
                with self.trace(span_name=span_or_func_name, tags=tags):
                    return func(*args, **kwargs)
            return tf_wrapper

    def profile_function(self, func, category: Optional[str] = None, event_name: Optional[str] = None):
        self._function_profiler.add_function(func, category, event_name)

    def profile_function_path(self, path, category: Optional[str] = None, event_name: Optional[str] = None):
        self._function_profiler.add_function_path(path, category, event_name)

    def profile_cuda_kernel(self, kernel_pattern: str, event_name: str):
        self._cupti_profiler.add_kernel_pattern(kernel_pattern,event_name)

    def set_gauge(self, name, value, measurement_ts, unit=None, aggregate=False, tags=None):
        self._metric_store.set_gauge(name=name, value=value, measurement_ts=measurement_ts, unit=unit, aggregate=aggregate, tags=tags)

    def inc_counter(self, name, value, measurement_ts, unit=None, aggregate=False, tags=None):
        self._metric_store.inc_counter(name=name, value=value, measurement_ts=measurement_ts, unit=unit, aggregate=aggregate, tags=tags)

    def update_summary(self, name, count, sum_val, sum2_val, measurement_ts, unit=None, aggregate=False, tags=None):
        self._metric_store.update_summary(name=name, count=count, sum_val=sum_val, sum2_val=sum2_val, measurement_ts=measurement_ts, unit=unit, aggregate=aggregate, tags=tags)

    def update_histogram(self, name, value, measurement_ts, unit=None, aggregate=False, tags=None):
        self._metric_store.update_histogram(name=name, value=value, measurement_ts=measurement_ts, unit=unit, aggregate=aggregate, tags=tags)

    def add_gauge_profile_field(self, descriptor):
        return self._metric_store.add_gauge_profile_field(descriptor)

    def add_counter_profile_field(self, descriptor):
        return self._metric_store.add_counter_profile_field(descriptor)

    def update_profile(self, name, profile, measurement_ts, unit=None, tags=None):
        self._metric_store.update_profile(name=name, profile=profile, measurement_ts=measurement_ts, unit=unit, tags=tags)

    def log_message(self, message: str, *, tags: Optional[Dict[str, str]] = None, level: Optional[str] = None, exception: Optional[str] = None):
        self.log_store().log_message(message=message, tags=tags, level=level, exception=exception)

    def tick(self, block=False, force=False):
        now = time.time()
        if not force and (now - self.last_tick_ts) < Ticker.TICK_INTERVAL_SEC - 1:
            return
        
        if not self._tick_lock.acquire(blocking=False):
            return

        try:
            def _run_tick():
                try:
                    try:
                        self.config_loader().update_config()
                    except Exception as exc:
                        logger.error('Error in config loader update: %s', exc, exc_info=True)

                    try:
                        self.emit_tick()
                    except Exception as exc:
                        logger.error('Error in metric read event handlers', exc_info=True)

                    if self._metric_store.has_unexported():
                        metrics = self._metric_store.export()
                        for metric in metrics:
                            self.signal_uploader().upload_metric(metric)

                    if self._log_store.has_unexported():
                        batches = self._log_store.export()
                        for batch in batches:
                            self.signal_uploader().upload_log_batch(batch)

                    self._signal_uploader.flush()
                except Exception as exc:
                    logger.error('Error in tick execution: %s', exc, exc_info=True)

            self.last_tick_ts = now

            self._tick_run_thread = threading.Thread(target=_run_tick, daemon=True)
            self._tick_run_thread.start()
            if block:
                self._tick_run_thread.join()
        finally:
            self._tick_lock.release()
