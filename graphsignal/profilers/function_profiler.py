import logging
import sys
import threading
import time
from pydoc import locate
import inspect
from inspect import isfunction

import graphsignal

logger = logging.getLogger('graphsignal')


class FunctionFields:
    __slots__ = ['duration_field_id', 'calls_field_id', 'errors_field_id']

    def __init__(self, duration_field_id=None, calls_field_id=None, errors_field_id=None):
        self.duration_field_id = duration_field_id
        self.calls_field_id = calls_field_id
        self.errors_field_id = errors_field_id

class FunctionBucket:
    __slots__ = [
        'bucket_ts', 
        'num_running', 
        'num_exited', 
        'num_errors', 
        'enter_offset_ns', 
        'exit_offset_ns'
    ]

    def __init__(self):
        self.bucket_ts = 0
        self.num_running = 0
        self.num_exited = 0
        self.num_errors = 0
        self.enter_offset_ns = 0
        self.exit_offset_ns = 0
        
    def enter(self):
        now_ns = time.time_ns()
        self.enter_offset_ns += now_ns - self.bucket_ts
        self.num_running += 1

    def exit(self, exc=None):
        now_ns = time.time_ns()
        self.exit_offset_ns += now_ns - self.bucket_ts

        self.num_exited += 1
        if exc:
            self.num_errors += 1
        self.num_running -= 1

    def rollover(self, bucket_ts):
        self.bucket_ts = bucket_ts
        self.num_exited = 0
        self.num_errors = 0
        self.enter_offset_ns = 0
        self.exit_offset_ns = 0


class FunctionProfiler():
    def __init__(self, profile_name):
        self._profile_name = profile_name
        self._resolution_ns = 10_000_000  # 10ms default resolution
        self._disabled = True
        self._tool_id = None
        self._fields = {}
        self._buckets = {}
        self._bucket_lock = threading.Lock()
        self._current_bucket_ts = None
        self._rollover_stop_event = threading.Event()
        self._rollover_timer_thread = None
        
    def set_resolution_ns(self, resolution_ns):
        if resolution_ns < 10_000_000:
            resolution_ns = 10_000_000 # 10ms minimum resolution
        self._resolution_ns = resolution_ns
        
    def get_resolution_ns(self):
        return self._resolution_ns
    
    def setup(self):
        try:
            mon = sys.monitoring
        except ImportError:
            logger.debug('sys.monitoring is not available')
            return

        self._tool_id = 4 # GRAPHSIGNAL_ID

        mon.use_tool_id(self._tool_id, "graphsignal-python-profiler")
        mon.register_callback(self._tool_id, mon.events.PY_START, self._py_start_callback)
        mon.register_callback(self._tool_id, mon.events.PY_RETURN, self._py_return_callback)
        mon.register_callback(self._tool_id, mon.events.PY_UNWIND, self._py_unwind_callback)

        self._start_rollover_timer()

        self._current_bucket_ts = time.time_ns()
        self._disabled = False

    def shutdown(self):
        if self._disabled:
            return

        sys.monitoring.free_tool_id(self._tool_id)
        
        self._stop_rollover_timer()

        self._buckets.clear()
        self._fields.clear()

    def _build_descriptor(self, func, category=None, event_name=None, statistic=None, unit=None):
        descriptor = {}

        code = getattr(func, "__code__", None)
        if code is not None:
            descriptor['filename'] = code.co_filename
            descriptor['lineno'] = code.co_firstlineno

        func_name = getattr(func, "__qualname__", None) or getattr(func, "__name__", None)
        if func_name:
            descriptor['function'] = func_name

        if category is not None:
            descriptor['category'] = category

        if event_name is not None:
            descriptor['event_name'] = event_name
        elif func_name:
            descriptor['event_name'] = func_name

        if statistic is not None:
            descriptor['statistic'] = statistic

        if unit is not None:
            descriptor['unit'] = unit

        return descriptor

    def add_function(self, func, category=None, event_name=None):
        if self._disabled:
            return

        if not hasattr(func, '__code__'):
            return

        if category is None:
            category = 'python'

        descriptor = self._build_descriptor(func, category, event_name, 'duration', unit='ns')
        duration_field_id = graphsignal._ticker.add_counter_profile_field(descriptor=descriptor)

        descriptor = self._build_descriptor(func, category, event_name, 'call_count')
        calls_field_id = graphsignal._ticker.add_counter_profile_field(descriptor=descriptor)

        descriptor = self._build_descriptor(func, category, event_name, 'error_count')
        errors_field_id = graphsignal._ticker.add_counter_profile_field(descriptor=descriptor)

        code = func.__code__
        self._fields[code] = FunctionFields(
            duration_field_id=duration_field_id,
            calls_field_id=calls_field_id,
            errors_field_id=errors_field_id
        )

        E = sys.monitoring.events
        sys.monitoring.set_local_events(self._tool_id, code, E.PY_START | E.PY_RETURN)
        sys.monitoring.set_events(self._tool_id, E.PY_UNWIND)

    def add_function_path(self, path, category=None, event_name=None):
        try:
            func = locate(path)
        except Exception as e:
            logger.debug(f"Could not resolve '{path}': {e}")
            return
        if func is None:
            logger.debug(f"Could not resolve '{path}': not found")
            return

        try:
            func = inspect.unwrap(func)
        except Exception:
            # If unwrap fails, keep the original object.
            pass

        if inspect.ismethod(func):
            func = func.__func__
        elif hasattr(func, "__func__") and isfunction(getattr(func, "__func__", None)):
            func = func.__func__

        if not hasattr(func, "__code__"):
            logger.debug(
                f"Could not resolve '{path}': not a Python function (no __code__)"
            )
            return
        self.add_function(func, category, event_name)
            
    def _py_start_callback(self, code, off):
        if self._disabled:
            return
        try:
            self._enter_callback(code, off)
        except Exception as e:
            logger.error(f"Error in _py_start_callback: {e}")
            return

    def _py_return_callback(self, code, off, retval):
        if self._disabled:
            return
        try:
            self._exit_callback(code)
        except Exception as e:
            logger.error(f"Error in _py_return_callback: {e}")
            return

    def _py_unwind_callback(self, code, off, exc):
        if self._disabled:
            return
        try:
            self._exit_callback(code, exc)
        except Exception as e:
            logger.error(f"Error in _py_unwind_callback: {e}")
            return

    def _enter_callback(self, code, off):
        fields = self._fields.get(code)
        if not fields:
            return

        bucket = self._buckets.get(code)
        if not bucket and self._fields.get(code):
            bucket = FunctionBucket()
            # buckets are created when the function is first seen, so there is practically no lock overhead here
            with self._bucket_lock:
                self._buckets[code] = bucket

        bucket.enter()

    def _exit_callback(self, code, exc=None):
        bucket = self._buckets.get(code)
        if not bucket:
            return

        bucket.exit(exc)

    def _start_rollover_timer(self):
        self._rollover_stop_event = threading.Event()

        def round_to_rollup(bucket_ts):
            return bucket_ts // self._resolution_ns * self._resolution_ns

        def _rollover_loop():
            while not self._rollover_stop_event.wait(self._resolution_ns / 1e9 / 10):
                try:
                    now_ns = time.time_ns()
                    if round_to_rollup(now_ns) > round_to_rollup(self._current_bucket_ts):
                        self._rollover_buckets(now_ns)
                except Exception as exc:
                    logger.error('Error in rollover timer: %s', exc, exc_info=True)

        self._rollover_timer_thread = threading.Thread(target=_rollover_loop, daemon=True)
        self._rollover_timer_thread.start()

    def _stop_rollover_timer(self):
        if self._rollover_timer_thread:
            self._rollover_stop_event.set()
            self._rollover_timer_thread.join()
            self._rollover_stop_event = None
            self._rollover_timer_thread = None

    def _rollover_buckets(self, now_ns):
        if self._disabled:
            return

        bucket_size_ns = now_ns - self._current_bucket_ts

        profile = {}
        with self._bucket_lock:
            for code, bucket in self._buckets.items():
                fields = self._fields.get(code)
                if not fields:
                    continue
                if bucket.num_running > 0 or bucket.exit_offset_ns > 0:
                    duration = bucket_size_ns * bucket.num_running - bucket.enter_offset_ns + bucket.exit_offset_ns
                    duration = max(0, duration)
                    if fields.duration_field_id and duration > 0:
                        profile[fields.duration_field_id] = duration
                    num_calls = bucket.num_running + bucket.num_exited
                    if fields.calls_field_id and num_calls > 0:
                        profile[fields.calls_field_id] = num_calls
                    if fields.errors_field_id and bucket.num_errors > 0:
                        profile[fields.errors_field_id] = bucket.num_errors

                bucket.rollover(now_ns)

        if len(profile) > 0:
            graphsignal._ticker.update_profile(name=self._profile_name, profile=profile, measurement_ts=now_ns)

        self._current_bucket_ts = now_ns
