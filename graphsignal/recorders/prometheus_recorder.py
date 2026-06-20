import logging
import math
import time
from typing import Optional

try:
    from prometheus_client.parser import text_string_to_metric_families
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

try:
    import urllib.request
    import urllib.error
    HTTP_AVAILABLE = True
except ImportError:
    HTTP_AVAILABLE = False

import graphsignal
import graphsignal.sdk
from graphsignal.recorders.base_recorder import BaseRecorder

logger = logging.getLogger('graphsignal')

INITIAL_DETECT_DELAY_SEC = 2.0
MAX_DETECT_DELAY_SEC = 60.0
DEFAULT_METRICS_PATH = '/metrics'
DEFAULT_METRICS_HOST = '127.0.0.1'


def _is_finite_number(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def normalize_metrics_path(path: Optional[str]) -> str:
    normalized = (path or DEFAULT_METRICS_PATH).strip()
    if not normalized.startswith('/'):
        normalized = '/' + normalized
    return normalized


def format_metrics_host(host: Optional[str]) -> str:
    host = (host or DEFAULT_METRICS_HOST).strip()
    if ':' in host and not host.startswith('['):
        return f'[{host}]'
    return host


def build_metrics_endpoint(metrics_port: int, metrics_path: Optional[str] = None,
                           metrics_host: Optional[str] = None) -> str:
    path = normalize_metrics_path(metrics_path)
    host = format_metrics_host(metrics_host)
    return f'http://{host}:{int(metrics_port)}{path}'


class PrometheusRecorder(BaseRecorder):
    """Scrapes a single, known Prometheus metrics HTTP endpoint.

    The scrape port is resolved by the launcher (from `--metrics-port` or the
    engine's serving port) and passed in explicitly. The path defaults to
    `/metrics` (vLLM, SGLang); TensorRT-LLM uses `/prometheus/metrics`.
    We never enumerate or probe a process's other listening sockets —
    blindly connecting to them can corrupt internal IPC channels (e.g.
    TensorRT-LLM's ZeroMQ queues).
    """

    def __init__(self, pid=None, args=None, metrics_port=None,
                 metrics_path: Optional[str] = None,
                 metrics_host: Optional[str] = None):
        super().__init__(pid=pid, args=args)
        self._endpoint: Optional[str] = None
        if metrics_port is not None:
            self._endpoint = build_metrics_endpoint(
                metrics_port, metrics_path=metrics_path, metrics_host=metrics_host)
        self._verified: bool = False
        self._last_values: dict = {}
        self._next_detect_ts: float = 0.0
        self._detect_delay_sec: float = INITIAL_DETECT_DELAY_SEC

    def setup(self):
        # Scraping is lazy; the first on_tick waits the initial delay since the
        # server may not be listening yet right after launch.
        self._next_detect_ts = time.time() + INITIAL_DETECT_DELAY_SEC

    def on_tick(self):
        if not PROMETHEUS_AVAILABLE or not HTTP_AVAILABLE:
            return
        if self._endpoint is None:
            return

        if not self._verified and time.time() < self._next_detect_ts:
            return

        try:
            body = self._fetch_metrics(self._endpoint)
        except Exception as exc:
            logger.debug('Failed to fetch %s: %s', self._endpoint, exc)
            # Server may still be starting up; back off and retry the same port.
            self._verified = False
            self._detect_delay_sec = min(self._detect_delay_sec * 2, MAX_DETECT_DELAY_SEC)
            self._next_detect_ts = time.time() + self._detect_delay_sec
            return

        if not self._verified:
            if not _looks_like_prometheus(body):
                self._detect_delay_sec = min(self._detect_delay_sec * 2, MAX_DETECT_DELAY_SEC)
                self._next_detect_ts = time.time() + self._detect_delay_sec
                return
            self._verified = True
            logger.debug('Prometheus /metrics endpoint confirmed: %s', self._endpoint)

        try:
            self._parse_and_emit(body)
        except Exception as exc:
            logger.error('Failed to parse Prometheus metrics: %s', exc, exc_info=True)

    @staticmethod
    def _fetch_metrics(url: str, timeout: float = 2.0) -> str:
        req = urllib.request.Request(url, headers={'Accept': 'text/plain'})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
            charset = resp.headers.get_content_charset() or 'utf-8'
            return data.decode(charset, errors='replace')

    def _parse_and_emit(self, body: str) -> None:
        sdk = graphsignal.sdk.sdk()
        now_ns = time.time_ns()

        for family in text_string_to_metric_families(body):
            name = family.name
            mtype = family.type

            sample_groups = {}
            for sample in family.samples:
                labels = {k: v for k, v in sample.labels.items() if k not in ('le', 'quantile')}
                group_key = frozenset(labels.items())
                sample_groups.setdefault(group_key, {})[sample.name] = sample

            for group_key, sample_map in sample_groups.items():
                tags = dict(group_key)

                if mtype == 'gauge':
                    s = sample_map.get(name)
                    if s is not None and _is_finite_number(s.value):
                        sdk.set_gauge(name=name, tags=tags, value=s.value, measurement_ts=now_ns)

                elif mtype == 'counter':
                    s = sample_map.get(f'{name}_total') or sample_map.get(name)
                    if s is not None and _is_finite_number(s.value):
                        last_key = (name, group_key)
                        current = s.value
                        prev = self._last_values.get(last_key)
                        if prev is not None:
                            delta = current - prev
                            if delta >= 0:
                                sdk.inc_counter(name=name, tags=tags, value=delta, measurement_ts=now_ns)
                        self._last_values[last_key] = current

                elif mtype in ('histogram', 'summary'):
                    c = sample_map.get(f'{name}_count')
                    su = sample_map.get(f'{name}_sum')
                    if c is not None and su is not None and _is_finite_number(c.value) and _is_finite_number(su.value):
                        last_key = (name, group_key)
                        cur_c, cur_s = c.value, su.value
                        prev = self._last_values.get(last_key)
                        if prev is not None:
                            dc, ds = cur_c - prev[0], cur_s - prev[1]
                            if dc > 0 and _is_finite_number(ds):
                                sdk.update_summary(name=name, tags=tags,
                                                   count=int(dc), sum_val=ds, measurement_ts=now_ns)
                        self._last_values[last_key] = (cur_c, cur_s)


def _looks_like_prometheus(body: str) -> bool:
    if not body:
        return False
    # OpenMetrics/Prometheus payloads begin with HELP/TYPE comments or metric samples.
    for line in body.splitlines():
        if not line:
            continue
        if line.startswith('# HELP') or line.startswith('# TYPE'):
            return True
        if line.startswith('#'):
            continue
        if ' ' in line and not line.startswith('<'):
            return True
        return False
    return False
