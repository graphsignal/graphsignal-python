from typing import Dict, Optional
import hashlib
import logging
import os
import time
import threading
import uuid

from graphsignal.sdk.signal_uploader import SignalUploader
from graphsignal.sdk.config_loader import ConfigLoader
from graphsignal.sdk.pid_watcher import PidWatcher
from graphsignal.signals.metrics import MetricStore
from graphsignal.signals.logs import LogStore
from graphsignal.signals.resources import ResourceStore
from graphsignal.signals.spans import SpanStore
from graphsignal.profilers.span_profiler import SpanProfiler
from graphsignal.otel.otel_collector import OTELCollector

logger = logging.getLogger('graphsignal')


def uuid_sha1(size=-1):
    sha1_hash = hashlib.sha1()
    sha1_hash.update(str(uuid.uuid4()).encode('utf-8'))
    return sha1_hash.hexdigest()[0:size]


class GraphsignalLogHandler(logging.Handler):
    def __init__(self, sdk):
        super().__init__()
        self._sdk = sdk

    def emit(self, record):
        try:
            exception = None
            if record.exc_info and isinstance(record.exc_info, tuple):
                exception = self.format(record)

            self._sdk.log_store().log_sdk_message(
                level=record.levelname,
                message=record.getMessage(),
                exception=exception)
        except Exception:
            pass


class Sdk:
    TICK_DELAY_SEC = 1
    TICK_INTERVAL_SEC = 1
    MAX_TAGS = 25

    def __init__(
            self,
            api_key=None,
            api_base=None,
            tags=None,
            debug_mode=False,
            target_pid=None,
            otel_collector_port=None,
            metrics_port=None):
        if debug_mode:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.WARNING)

        if not api_key:
            raise ValueError('api_key is required')

        self._api_key = api_key
        if api_base:
            self._api_base = api_base
        else:
            self._api_base = 'https://api.graphsignal.com'
        self._tags = {}
        if tags:
            self._tags.update(tags)

        self.debug_mode = debug_mode

        self._target_pid = int(target_pid) if target_pid is not None else os.getpid()
        self._otel_collector_port = int(otel_collector_port) if otel_collector_port is not None else None
        self._metrics_port = int(metrics_port) if metrics_port is not None else None

        self._tick_timer_thread = None
        self._tick_stop_event = threading.Event()
        self._tick_lock = threading.Lock()
        self._tick_run_thread = None
        self._python_log_handler = None
        self._config_loader = None
        self._signal_uploader = None
        self._metric_store = None
        self._log_store = None
        self._resource_store = None
        self._span_store = None
        self._span_profiler = None

        self._otel_collector = None
        self._pid_watcher = None
        self._global_recorders: list = []
        self._child_recorders: dict = {}
        self._recorders_lock = threading.Lock()

        self._target_terminated_event = threading.Event()
        self._shutdown_started = False
        self._shutdown_lock = threading.Lock()

        self._process_start_ms = int(time.time() * 1e3)
        self._last_tick_ts = time.time()
        self._auto_tick = True

    def setup(self):
        self._log_store = LogStore()
        if not self._python_log_handler:
            self._python_log_handler = GraphsignalLogHandler(self)
            logger.addHandler(self._python_log_handler)

        logger.info('SDK started: pid=%s', os.getpid())
        logger.debug('SDK setup started')

        self._config_loader = ConfigLoader()
        self._config_loader.setup()

        def update_func(changed_options):
            if 'traces_per_sec' in changed_options and self._span_store is not None:
                self._span_store.reset_samplers()
        self._config_loader.on_update(update_func)

        self._signal_uploader = SignalUploader()
        self._signal_uploader.setup()

        self._metric_store = MetricStore()
        self._resource_store = ResourceStore()
        self._span_store = SpanStore(config_loader=self._config_loader)

        self._span_profiler = SpanProfiler(profile_name='profile.events')
        self._span_profiler.setup()

        if self._otel_collector_port is not None:
            logger.info('Starting OTEL collector on port %s', self._otel_collector_port)
            self._otel_collector = OTELCollector(port=self._otel_collector_port)
            self._otel_collector.setup()
        else:
            logger.info('No OTEL collector port configured; OTEL traces disabled '
                        '(no --otel-collector-port; the launcher allocates one only '
                        'when no user --otlp-traces-endpoint is supplied)')

        self._pid_watcher = PidWatcher(self._target_pid)
        self._pid_watcher.add_listener(self)
        self._pid_watcher.setup()

        self._start_tick_timer()

        if hasattr(os, 'register_at_fork'):
            os.register_at_fork(after_in_child=self._shutdown_in_fork_child)

        logger.debug('SDK setup complete')

    def on_target_created(self, args):
        from graphsignal.recorders.host_recorder import HostRecorder
        from graphsignal.recorders.process_recorder import ProcessRecorder
        from graphsignal.recorders.nvml_recorder import NVMLRecorder
        from graphsignal.recorders.cupti_recorder import CuptiRecorder
        from graphsignal.recorders.prometheus_recorder import PrometheusRecorder

        recorders = []
        recorders.append(HostRecorder(pid=self._target_pid, args=args))
        recorders.append(NVMLRecorder(pid=self._target_pid, args=args))
        if self._metrics_port is not None:
            recorders.append(PrometheusRecorder(
                pid=self._target_pid, args=args, metrics_port=self._metrics_port))
        recorders.append(ProcessRecorder(pid=self._target_pid, args=args))
        recorders.append(CuptiRecorder(pid=self._target_pid, args=args))

        with self._recorders_lock:
            self._global_recorders = recorders

        for recorder in recorders:
            try:
                recorder.setup()
            except Exception:
                logger.error('Failed to set up recorder %s for target pid %s',
                             type(recorder).__name__, self._target_pid, exc_info=True)

    def on_child_created(self, pid, args):
        from graphsignal.recorders.process_recorder import ProcessRecorder
        from graphsignal.recorders.cupti_recorder import CuptiRecorder

        recorders = [
            ProcessRecorder(pid=pid, args=args),
            CuptiRecorder(pid=pid, args=args),
        ]
        with self._recorders_lock:
            self._child_recorders[pid] = recorders

        for recorder in recorders:
            try:
                recorder.setup()
            except Exception:
                logger.error('Failed to set up recorder %s for child pid %s',
                             type(recorder).__name__, pid, exc_info=True)

    def on_child_terminated(self, pid):
        with self._recorders_lock:
            recorders = self._child_recorders.pop(pid, [])
        for recorder in recorders:
            try:
                recorder.shutdown()
            except Exception:
                logger.error('Failed to shutdown recorder %s for child pid %s',
                             type(recorder).__name__, pid, exc_info=True)

    def on_target_terminated(self):
        # Run a final flush + shutdown on a background thread so the watcher loop
        # can return promptly; signal target_terminated_event after shutdown completes.
        def _finalize():
            try:
                self._auto_tick = False
                self.tick(block=True, force=True)
            except Exception:
                logger.error('Error during target_terminated final flush', exc_info=True)
            finally:
                self._target_terminated_event.set()

        threading.Thread(target=_finalize, daemon=True).start()

    def _start_tick_timer(self):
        self._tick_stop_event = threading.Event()

        def _tick_loop():
            if not self._tick_stop_event.wait(Sdk.TICK_DELAY_SEC):
                try:
                    if self._auto_tick:
                        self.tick(force=True)
                except Exception as exc:
                    logger.error('Error in initial tick: %s', exc, exc_info=True)

            while not self._tick_stop_event.wait(Sdk.TICK_INTERVAL_SEC):
                try:
                    if self._auto_tick:
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

    def _shutdown_in_fork_child(self):
        # The SDK runs only in the parent. In a forked child, shut everything
        # down via the module-level shutdown() so the singleton is cleared as
        # well (otherwise `is_configured()` would still report True).
        try:
            import graphsignal.sdk as gsdk
            gsdk.shutdown()
        except Exception:
            pass

    def shutdown(self):
        with self._shutdown_lock:
            if self._shutdown_started:
                return
            self._shutdown_started = True

        if self._auto_tick:
            try:
                self.tick(block=True, force=True)
            except Exception:
                logger.error('Error in final tick during shutdown', exc_info=True)

        if self._tick_stop_event:
            self._tick_stop_event.set()

        if self._tick_timer_thread:
            try:
                self._tick_timer_thread.join(timeout=5.0)
            except Exception:
                pass
            self._tick_timer_thread = None

        if self._tick_run_thread:
            try:
                self._tick_run_thread.join(timeout=5.0)
            except Exception:
                pass
            self._tick_run_thread = None

        if self._pid_watcher:
            try:
                self._pid_watcher.shutdown()
            except Exception:
                pass
            self._pid_watcher = None

        for recorder in self.recorders():
            try:
                recorder.shutdown()
            except Exception:
                logger.error('Error shutting down recorder', exc_info=True)
        with self._recorders_lock:
            self._global_recorders = []
            self._child_recorders = {}

        if self._otel_collector:
            try:
                self._otel_collector.shutdown()
            except Exception:
                pass
            self._otel_collector = None

        if self._span_profiler:
            try:
                self._span_profiler.shutdown()
            except Exception:
                pass
            self._span_profiler = None

        self._metric_store = None
        self._log_store = None
        self._resource_store = None
        self._span_store = None
        self._signal_uploader = None

        if self._config_loader:
            try:
                self._config_loader.shutdown()
            except Exception:
                pass
            self._config_loader = None

        self._tags = None

        if self._python_log_handler:
            logger.removeHandler(self._python_log_handler)
            self._python_log_handler = None

    def target_pid(self) -> int:
        return self._target_pid

    def otel_collector(self):
        return self._otel_collector

    def target_terminated_event(self) -> threading.Event:
        return self._target_terminated_event

    def config_loader(self):
        return self._config_loader

    def signal_uploader(self):
        return self._signal_uploader

    def span_profiler(self):
        return self._span_profiler

    def metric_store(self):
        return self._metric_store

    def log_store(self):
        return self._log_store

    def resource_store(self):
        return self._resource_store

    def span_store(self):
        return self._span_store

    def recorders(self):
        with self._recorders_lock:
            global_recs = list(self._global_recorders)
            child_recs = [r for rs in self._child_recorders.values() for r in rs]
        yield from global_recs
        yield from child_recs

    def tags(self) -> Dict[str, str]:
        if self._tags is None:
            return {}
        return self._tags.copy()

    def api_key(self) -> str:
        return self._api_key

    def api_base(self) -> str:
        return self._api_base

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
            self._tags.pop(key, None)
            return

        if len(self._tags) > Sdk.MAX_TAGS:
            logger.error('set_tag: too many tags (>{0})'.format(Sdk.MAX_TAGS))
            return

        if append_uuid:
            if not value:
                value = uuid_sha1(size=12)
            else:
                value = '{0}-{1}'.format(value, uuid_sha1(size=12))

        self._tags[key] = value

    def get_tag(self, key: str) -> Optional[str]:
        return self._tags.get(key, None)

    def remove_tag(self, key: str) -> None:
        self._tags.pop(key, None)

    def set_gauge(self, name, value, measurement_ts, unit=None, aggregate=False, tags=None):
        self._metric_store.set_gauge(name=name, value=value, measurement_ts=measurement_ts, unit=unit, aggregate=aggregate, tags=tags)

    def inc_counter(self, name, value, measurement_ts, unit=None, aggregate=False, tags=None):
        self._metric_store.inc_counter(name=name, value=value, measurement_ts=measurement_ts, unit=unit, aggregate=aggregate, tags=tags)

    def update_summary(self, name, count, sum_val, sum2_val=0, measurement_ts=None, unit=None, aggregate=False, tags=None):
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

    def update_resource(self, kind, tags=None, attributes=None, first_seen_ts=None, last_seen_ts=None):
        self._resource_store.update_resource(kind=kind, tags=tags, attributes=attributes, first_seen_ts=first_seen_ts, last_seen_ts=last_seen_ts)

    def record_span(self, *, trace_id, span_id, name, start_ts, end_ts,
                    parent_span_id=None, attributes=None, events=None,
                    tags=None):
        return self._span_store.record_span(
            trace_id=trace_id,
            span_id=span_id,
            name=name,
            start_ts=start_ts,
            end_ts=end_ts,
            parent_span_id=parent_span_id,
            attributes=attributes,
            events=events,
            tags=tags)

    def tick(self, block=False, force=False):
        now = time.time()
        if not force and (now - self._last_tick_ts) < Sdk.TICK_INTERVAL_SEC - 1:
            return

        if not self._tick_lock.acquire(blocking=False):
            return

        try:
            def _run_tick():
                try:
                    try:
                        if self._config_loader:
                            self.config_loader().update_config()
                    except Exception as exc:
                        logger.error('Error in config loader update: %s', exc, exc_info=True)

                    try:
                        self.emit_tick()
                    except Exception:
                        logger.error('Error in tick recorder loop', exc_info=True)

                    if self._span_store and self._span_store.has_unexported():
                        spans = self._span_store.export()
                        for span in spans:
                            self.signal_uploader().upload_span(span)

                    if self._metric_store and self._metric_store.has_unexported():
                        metrics = self._metric_store.export()
                        for metric in metrics:
                            self.signal_uploader().upload_metric(metric)

                    if self._log_store and self._log_store.has_unexported():
                        batches = self._log_store.export()
                        for batch in batches:
                            self.signal_uploader().upload_log_batch(batch)

                    if self._resource_store and self._resource_store.has_unexported():
                        resources = self._resource_store.export()
                        for resource in resources:
                            self.signal_uploader().upload_resource(resource)

                    if self._signal_uploader:
                        self._signal_uploader.flush()
                except Exception as exc:
                    logger.error('Error in tick execution: %s', exc, exc_info=True)

            self._last_tick_ts = now

            self._tick_run_thread = threading.Thread(target=_run_tick, daemon=True)
            self._tick_run_thread.start()
            if block:
                self._tick_run_thread.join()
        finally:
            self._tick_lock.release()
