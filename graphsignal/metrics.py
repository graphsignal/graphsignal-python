import logging
import sys
import threading
import time
import math

import graphsignal
from graphsignal import version
from graphsignal.proto import signals_pb2

logger = logging.getLogger('graphsignal')


class BaseMetric:
    def __init__(self, scope, name, tags, unit=None, is_time=False, is_size=False):
        self._update_lock = threading.Lock()
        self.scope = scope
        self.name = name
        self.tags = tags
        self.unit = unit
        self.is_time = is_time
        self.is_size = is_size
        self.is_updated = False

    def update(self):
        self.is_updated = True

    def export(self):
        self.is_updated = False

        proto = signals_pb2.Metric()
        proto.scope = self.scope
        proto.name = self.name
        for key, value in self.tags.items():
            tag = proto.tags.add()
            tag.key = str(key)[:50]
            tag.value = str(value)[:250]
        proto.type = self.type
        if self.unit is not None:
            proto.unit = self.unit
        proto.is_time = self.is_time
        proto.is_size = self.is_size
        proto.update_ts = self.update_ts
        return proto


class GaugeMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.type = signals_pb2.Metric.MetricType.GAUGE_METRIC
        self.gauge = None

    def update(self, value, update_ts):
        with self._update_lock:
            super().update()
            self.gauge = value
            self.update_ts = update_ts

    def export(self):
        with self._update_lock:
            proto = super().export()
            if self.gauge is not None:
                proto.gauge = self.gauge
            self.gauge = None
            return proto


class CounterMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.type = signals_pb2.Metric.MetricType.COUNTER_METRIC
        self.counter = 0

    def update(self, value, update_ts):
        with self._update_lock:
            super().update()
            self.counter += value
            self.update_ts = update_ts

    def export(self):
        with self._update_lock:
            proto = super().export()
            if self.counter > 0:
                proto.counter = self.counter
                self.counter = 0
            return proto


class HistogramMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.type = signals_pb2.Metric.MetricType.HISTOGRAM_METRIC
        self.histogram = {}

    def update(self, value, update_ts):
        if value < 1:
            return
        with self._update_lock:
            super().update()
            bin_size = max(10 ** (int(math.log(value, 10)) - 1), 1)
            bin = int(value / bin_size) * bin_size
            self.histogram[bin] = self.histogram.get(bin, 0) + 1
            self.update_ts = update_ts

    def export(self):
        with self._update_lock:
            proto = super().export()
            for bin, count in self.histogram.items():
                proto.histogram.bins.append(bin)
                proto.histogram.counts.append(count)
            self.histogram = {}
            return proto


class MetricStore:
    def __init__(self):
        self._update_lock = threading.Lock()
        self._has_unexported = False
        self._metrics = {}

    def metric_key(self, scope, name, tags):
        return (scope, name, frozenset(tags.items()))

    def set_gauge(self, scope, name, tags, value, update_ts, unit=None, is_time=False, is_size=False):
        key = self.metric_key(scope, name, tags)
        with self._update_lock:
            if key not in self._metrics:
                metric = self._metrics[key] = GaugeMetric(scope, name, tags, unit=unit, is_time=is_time, is_size=is_size)
            else:
                metric = self._metrics[key]
        metric.update(value, update_ts)
        return self._metrics[key]

    def inc_counter(self, scope, name, tags, value, update_ts, unit=None):
        key = self.metric_key(scope, name, tags)
        with self._update_lock:
            if key not in self._metrics:
                metric = self._metrics[key] = CounterMetric(scope, name, tags, unit=unit)
            else:
                metric = self._metrics[key]
        metric.update(value, update_ts)
        return self._metrics[key]

    def update_histogram(self, scope, name, tags, value, update_ts, unit=None, is_time=False, is_size=False):
        key = self.metric_key(scope, name, tags)
        with self._update_lock:
            if key not in self._metrics:
                metric = self._metrics[key] = HistogramMetric(scope, name, tags, unit=unit, is_time=is_time, is_size=is_size)
            else:
                metric = self._metrics[key]
        metric.update(value, update_ts)
        return self._metrics[key]

    def has_unexported(self):
        with self._update_lock:
            for metric in self._metrics.values():
                if metric.is_updated:
                    return True
        return False

    def export(self):
        protos = []
        with self._update_lock:
            for metric in self._metrics.values():
                if metric.is_updated:
                    protos.append(metric.export())
        return protos

    def clear(self):
        with self._update_lock:
            self._metrics = {}