import logging
import time

from graphsignal.windows import Metric
from graphsignal import system

logger = logging.getLogger('graphsignal')


class Span(object):
    __slots__ = [
        '_session',
        '_start_time',
        '_start_cpu_time',
        '_stopped'
    ]

    def __init__(self, session):
        self._session = session
        self._start_time = None
        self._stopped = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

        if exc_type and exc_val and exc_tb:
            self._session.log_event(
                description='Prediction exception',
                exc_info=(exc_type, exc_val, exc_tb)
            )

    def start(self):
        if self._stopped:
            return
        self._start_time = time.monotonic()
        self._start_cpu_time = system.cpu_time()

    def stop(self):
        if self._stopped:
            return
        self._stopped = True

        self._set_p50_metric(
            'prediction_latency_p50',
            round((time.monotonic() - self._start_time) * 1000))

        if self._start_cpu_time:
            self._set_p50_metric(
                'prediction_cpu_time_p50',
                round((system.cpu_time() - self._start_cpu_time) * 1000))

        self._session._set_updated()

    def _set_p50_metric(self, name, value):
        with self._session._update_lock:
            metric = None
            if name in self._session._metric_index:
                metric = self._session._metric_index[name]
            else:
                metric = self._session._metric_index[name] = Metric(
                    dataset=Metric.DATASET_SYSTEM,
                    name=name)
            metric.update_percentile(value, 50, Metric.UNIT_MILLISECOND)
