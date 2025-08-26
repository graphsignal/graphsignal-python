import logging
import sys
import threading
import time
import math

import graphsignal
from graphsignal import version
from graphsignal import client

logger = logging.getLogger('graphsignal')


class BaseMetric:
    def __init__(self, name, tags, unit=None, is_time=False, is_size=False):
        self._update_lock = threading.Lock()
        self.name = name
        self.tags = tags
        self.unit = unit
        self.is_time = is_time
        self.is_size = is_size
        self.is_updated = False

    def touch(self):
        self.is_updated = True

    def export(self):
        self.is_updated = False

        model = client.Metric(
            name=self.name,
            tags=[],
            type=self.type,
            is_time=self.is_time,
            is_size=self.is_size,
            update_ts=self.update_ts)
        for key, value in self.tags.items():
            model.tags.append(client.Tag(
                key=str(key)[:50],
                value=str(value)[:250]
            ))
        if self.unit is not None:
            model.unit = self.unit
        return model


class GaugeMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.type = 'gauge'
        self.gauge = None

    def update(self, value, update_ts):
        with self._update_lock:
            self.touch()
            self.gauge = value
            self.update_ts = update_ts

    def export(self):
        with self._update_lock:
            model = super().export()
            model.gauge = self.gauge
            self.gauge = None
            return model


class CounterMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.type = 'counter'
        self.total = 0

    def update(self, value, update_ts):
        with self._update_lock:
            self.touch()
            self.total += value
            self.update_ts = update_ts

    def export(self):
        with self._update_lock:
            model = super().export()
            model.total = self.total
            self.total = 0
            return model


class SummaryMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.type = 'summary'
        self.count = 0
        self.sum = 0
        self.sum2 = 0

    def update(self, count, sum_val, sum2_val, update_ts):
        with self._update_lock:
            self.touch()
            self.count += count
            self.sum += sum_val
            self.sum2 += sum2_val
            self.update_ts = update_ts

    def export(self):
        with self._update_lock:
            model = super().export()
            model.summary = client.Summary(count=int(self.count), sum=self.sum, sum2=self.sum2)
            return model


class HistogramMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.type = 'histogram'
        self.histogram = {}

    def update(self, value, update_ts):
        if value < 1:
            return
        with self._update_lock:
            self.touch()
            bin_size = max(10 ** (int(math.log(value, 10)) - 1), 1)
            bin = int(value / bin_size) * bin_size
            self.histogram[bin] = self.histogram.get(bin, 0) + 1
            self.update_ts = update_ts

    def export(self):
        with self._update_lock:
            model = super().export()
            model.histogram = client.Histogram(bins=[], counts=[])
            for bin, count in self.histogram.items():
                model.histogram.bins.append(bin)
                model.histogram.counts.append(int(count))
            self.histogram = {}
            return model


class MetricStore:
    def __init__(self):
        self._update_lock = threading.Lock()
        self._has_unexported = False
        self._metrics = {}

    def metric_key(self, name, tags):
        return (name, frozenset(tags.items()))

    def set_gauge(self, name, tags, value, update_ts, unit=None, is_time=False, is_size=False):
        if name is None:
            raise ValueError('Gauge name cannot be None')
        if value is None:
            raise ValueError('Gauge value cannot be None')

        key = self.metric_key(name, tags)
        with self._update_lock:
            if key not in self._metrics:
                metric = self._metrics[key] = GaugeMetric(name, tags, unit=unit, is_time=is_time, is_size=is_size)
            else:
                metric = self._metrics[key]
        metric.update(value, update_ts)
        return metric

    def inc_counter(self, name, tags, value, update_ts, unit=None):
        if name is None:
            raise ValueError('Counter name cannot be None')
        if value is None:
            raise ValueError('Counter value cannot be None')
        
        key = self.metric_key(name, tags)
        with self._update_lock:
            if key not in self._metrics:
                metric = self._metrics[key] = CounterMetric(name, tags, unit=unit)
            else:
                metric = self._metrics[key]
        metric.update(value, update_ts)
        return metric

    def update_summary(self, name, tags, count, sum_val, sum2_val, update_ts, unit=None):
        if name is None:
            raise ValueError('Summary name cannot be None')
        if count is None:
            raise ValueError('Summary count cannot be None')
        if sum_val is None:
            raise ValueError('Summary sum cannot be None')
        if sum2_val is None:
            raise ValueError('Summary sum2 cannot be None')
        
        key = self.metric_key(name, tags)
        with self._update_lock:
            if key not in self._metrics:
                metric = self._metrics[key] = SummaryMetric(name, tags, unit=unit)
            else:
                metric = self._metrics[key]
        metric.update(count, sum_val, sum2_val, update_ts)
        return metric

    def update_histogram(self, name, tags, value, update_ts, unit=None, is_time=False, is_size=False):
        if name is None:
            raise ValueError('Histogram name cannot be None')
        if value is None:
            raise ValueError('Histogram value cannot be None')

        key = self.metric_key(name, tags)
        with self._update_lock:
            if key not in self._metrics:
                metric = self._metrics[key] = HistogramMetric(name, tags, unit=unit, is_time=is_time, is_size=is_size)
            else:
                metric = self._metrics[key]
        metric.update(value, update_ts)
        return metric

    def has_unexported(self):
        with self._update_lock:
            for metric in self._metrics.values():
                if metric.is_updated:
                    return True
        return False

    def export(self):
        models = []
        with self._update_lock:
            for metric in self._metrics.values():
                if metric.is_updated:
                    models.append(metric.export())
        return models

    def clear(self):
        with self._update_lock:
            self._metrics = {}