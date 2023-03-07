import logging
import sys
import threading
import time
from collections import OrderedDict
import statistics

import graphsignal
from graphsignal import version
from graphsignal.proto import signals_pb2

logger = logging.getLogger('graphsignal')


class MetricStore:
    MAX_RESERVOIR_SIZE = 250
    MIN_SAMPLES = 30
    MAX_INTERVAL_SEC = 600

    def __init__(self):
        self._update_lock = threading.Lock()
        self.is_updated = False
        self.latency_reservoir = []
        self.latency_mean = 0
        self.latency_stdev = 0
        self.call_count = None
        self.exception_count = None
        self.data_counters = {}

    def add_latency(self, latency_us, timestamp_us):
        with self._update_lock:
            if len(self.latency_reservoir) < MetricStore.MAX_RESERVOIR_SIZE:
                self.latency_reservoir.append(latency_us)
            else:
                rand = timestamp_us % MetricStore.MAX_RESERVOIR_SIZE
                self.latency_reservoir[rand] = latency_us

        self.is_updated = True

    def is_latency_outlier(self, latency_us, timestamp_us):
        # Update mean and stdev approx every 10 samples to control overhead
        if (len(self.latency_reservoir) > MetricStore.MIN_SAMPLES and 
                (self.latency_mean == 0 or int(timestamp_us % 10) == 0)):
            self.latency_mean = statistics.mean(self.latency_reservoir)
            self.latency_stdev = statistics.stdev(self.latency_reservoir)

        # Samples worth investigating
        if self.latency_mean > 0:
            return self.latency_mean and latency_us > self.latency_mean + 6 * self.latency_stdev
        return False

    def inc_call_count(self, value, timestamp_us):
        with self._update_lock:
            if not self.call_count:
                self.call_count = OrderedDict()
            self._inc_counter(self.call_count, value, timestamp_us)
        
        self.is_updated = True

    def inc_exception_count(self, value, timestamp_us):
        with self._update_lock:
            if not self.exception_count:
                self.exception_count = OrderedDict()
            self._inc_counter(self.exception_count, value, timestamp_us)

        self.is_updated = True

    def inc_data_counter(self, data_name, counter_name, value, timestamp_us):
        with self._update_lock:
            key = (data_name, counter_name)
            if key not in self.data_counters:
                counter = OrderedDict()
                self.data_counters[key] = counter
            else:
                counter = self.data_counters[key]
            self._inc_counter(counter, value, timestamp_us)

        self.is_updated = True

    def _inc_counter(self, counter, value, timestamp_us):
        bucket = int(timestamp_us / 1e6)
        start_bucket = bucket - MetricStore.MAX_INTERVAL_SEC + 1

        has_expired = False
        for current_bucket in list(counter.keys()):
            if current_bucket < start_bucket:
                del counter[current_bucket]
                has_expired = True
            else:
                break
        # Any bucket expiration means the process is running for more than MAX_INTERVAL_SEC
        # Add a start bucket to properly compute rates in the backend
        if has_expired and start_bucket not in counter:
            counter[start_bucket] = 0

        counter[bucket] = counter.get(bucket, 0) + value

    def export(self, signal, timestamp_us):
        if self.latency_reservoir and len(self.latency_reservoir) > 0:
            self._convert_reservoir_to_proto(self.latency_reservoir, signal.trace_metrics.latency_us, timestamp_us)
        if self.call_count and len(self.call_count) > 0:
            self._convert_counter_to_proto(self.call_count, signal.trace_metrics.call_count, timestamp_us)
        if self.exception_count and len(self.exception_count) > 0:
            self._convert_counter_to_proto(self.exception_count, signal.trace_metrics.exception_count, timestamp_us)
        for (data_name, counter_name), data_counter in self.data_counters.items():
            data_metric = signal.data_metrics.add()
            data_metric.data_name = data_name
            data_metric.metric_name = counter_name
            self._convert_counter_to_proto(data_counter, data_metric.metric, timestamp_us)
        self.is_updated = False

    def _convert_reservoir_to_proto(self, reservoir, metric, timestamp_us):
        metric.type = signals_pb2.Metric.MetricType.RESERVOIR_METRIC
        for value in reservoir:
            metric.reservoir.values.append(value)

    def _convert_counter_to_proto(self, counter, metric, timestamp_us):
        metric.type = signals_pb2.Metric.MetricType.GAUGE_METRIC
        interval_sec = int(timestamp_us / 1e6) - min(counter.keys()) + 1
        metric.gauge = sum(counter.values()) / interval_sec

