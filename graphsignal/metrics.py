import copy
import logging
import math
import time
import hashlib
import random
import numpy as np

from graphsignal.sketches.kll import KLLSketch

logger = logging.getLogger('graphsignal')


def get_data_stream(window_proto, data_source, data_type):
    data_stream = window_proto.data_streams[str(data_source)]
    if data_stream.data_source == data_stream.DataSource.NOT_INITIALIZED:
        data_stream.data_source = data_source
        data_stream.data_type = data_type
    return data_stream


def get_metric_updater(
        metric_updaters, data_stream_proto, name, dimensions=None):
    metric_id = _sha1('{0}:{1}:{2}'.format(
        data_stream_proto.data_source,
        name,
        str(dict(sorted(dimensions.items()))) if dimensions is not None else ''),
        size=12)

    if metric_id not in metric_updaters:
        metric = metric_updaters[metric_id] = DataMetricUpdater(
            data_stream_proto.metrics[metric_id], name, dimensions)
        return metric
    else:
        return metric_updaters[metric_id]


class DataMetricUpdater(object):
    __slots__ = [
        '_metric',
        '_sketch'
    ]

    def __init__(self, metric, name, dimensions=None):
        self._metric = metric
        self._sketch = None

        self._metric.name = name
        if dimensions is not None:
            for name, value in dimensions.items():
                self._metric.dimensions[name] = value

    def update_gauge(self, value):
        if self._metric.type == self._metric.NOT_INITIALIZED:
            self._metric.type = self._metric.GAUGE
        self._metric.gauge_value.gauge = value

    def update_counter(self, value):
        if self._metric.type == self._metric.NOT_INITIALIZED:
            self._metric.type = self._metric.COUNTER
            self._metric.counter_value.counter = 0
        self._metric.counter_value.counter += value

    def update_ratio(self, value, total):
        if self._metric.type == self._metric.NOT_INITIALIZED:
            self._metric.type = self._metric.RATIO
            self._metric.ratio_value.counter = 0
            self._metric.ratio_value.total = 0
        self._metric.ratio_value.counter += value
        self._metric.ratio_value.total += total

    def update_distribution(self, values):
        if len(values) == 0:
            return

        if self._metric.type == self._metric.NOT_INITIALIZED:
            self._metric.type = self._metric.DISTRIBUTION
            self._metric.distribution_value.sketch_impl = self._metric.distribution_value.KLL10
            k_val = 128 if isinstance(values[0], (int, float)) else 10
            self._sketch = KLLSketch(k=k_val)

        for value in values:
            self._sketch.update(value)

    def finalize(self):
        if self._metric.type == self._metric.DISTRIBUTION:
            self._sketch.to_proto(self._metric.distribution_value.sketch_kll10)


def _sha1(text, size=-1):
    sha1_hash = hashlib.sha1()
    sha1_hash.update(text.encode('utf-8'))
    return sha1_hash.hexdigest()[0:size]
