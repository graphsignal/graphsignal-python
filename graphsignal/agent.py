import logging
import sys
import threading
import random
import time
import uuid
import hashlib

import graphsignal
from graphsignal.uploader import Uploader
from graphsignal.usage.process_reader import ProcessReader
from graphsignal.usage.nvml_reader import NvmlReader
from graphsignal.trace_sampler import TraceSampler
from graphsignal.proto import signals_pb2
from graphsignal.data.missing_value_detector import MissingValueDetector

logger = logging.getLogger('graphsignal')


class Agent:
    def __init__(self, api_key, deployment=None, debug_mode=False):
        self.worker_id = _uuid_sha1(size=12)
        self.api_key = api_key
        self.deployment = deployment
        self.debug_mode = debug_mode

        self._uploader = None
        self._process_reader = None
        self._nvml_reader = None
        self._trace_samplers = None
        self._metric_store = None
        self._supported_profiler = None
        self._mv_detector = None

    def start(self):
        self._uploader = Uploader()
        self._uploader.setup()
        self._process_reader = ProcessReader()
        self._process_reader.setup()
        self._nvml_reader = NvmlReader()
        self._nvml_reader.setup()
        self._trace_samplers = {}
        self._metric_store = {}
        self._mv_detector = MissingValueDetector()

    def stop(self):
        self.upload(block=True)
        self._process_reader.shutdown()
        self._nvml_reader.shutdown()
        self._trace_samplers = None
        self._metric_store = None
        self._supported_profiler = None

    def uploader(self):
        return self._uploader

    def supported_profiler(self):
        if self._supported_profiler:
            return self._supported_profiler

        if _check_module('torch'):
            from graphsignal.profilers.pytorch import PyTorchProfiler
            self._supported_profiler = PyTorchProfiler()
        elif _check_module('tensorflow'):
            from graphsignal.profilers.tensorflow import TensorFlowProfiler
            self._supported_profiler = TensorFlowProfiler()
        elif _check_module('jax'):
            from graphsignal.profilers.jax import JaxProfiler
            self._supported_profiler = JaxProfiler()
        else:
            from graphsignal.profilers.python import PythonProfiler
            self._supported_profiler = PythonProfiler()

        return self._supported_profiler
        
    def trace_sampler(self, endpoint):
        if endpoint in self._trace_samplers:
            return self._trace_samplers[endpoint]
        else:
            trace_sampler = self._trace_samplers[endpoint] = TraceSampler()
            return trace_sampler

    def reset_metric_store(self, endpoint):
        self._metric_store[endpoint] = MetricStore()

    def metric_store(self, endpoint):
        if endpoint not in self._metric_store:
            self.reset_metric_store(endpoint)

        return self._metric_store[endpoint]

    def mv_detector(self):
        return self._mv_detector

    def read_usage(self, signal):
        self._process_reader.read(signal)
        self._nvml_reader.read(signal)

    def create_signal(self):
        signal = signals_pb2.WorkerSignal()
        signal.worker_id = graphsignal._agent.worker_id
        signal.signal_id = _uuid_sha1(size=12)
        if self.deployment:
            signal.deployment_name = self.deployment
        return signal

    def upload(self, block=False):
        if block:
            self._uploader.flush()
        else:
            self._uploader.flush_in_thread()

    def tick(self, block=False):
        self.upload(block=False)


class MetricStore:
    MAX_RESERVOIR_SIZE = 100

    def __init__(self):
        self._update_lock = threading.Lock()
        self._start_sec = int(time.time())
        self.latency_us = None
        self.call_count = None
        self.exception_count = None
        self.data_counters = {}

    def add_time(self, duration_us):
        with self._update_lock:
            if not self.latency_us:
                self.latency_us = signals_pb2.Metric()
                self.latency_us.type = signals_pb2.Metric.MetricType.RESERVOIR_METRIC

            if len(self.latency_us.reservoir.values) < MetricStore.MAX_RESERVOIR_SIZE:
                self.latency_us.reservoir.values.append(duration_us)
            else:
                self.latency_us.reservoir.values[random.randint(0, MetricStore.MAX_RESERVOIR_SIZE - 1)] = duration_us

    def inc_call_count(self, value, timestamp_us):
        with self._update_lock:
            if not self.call_count:
                self.call_count = signals_pb2.Metric()
                self.call_count.type = signals_pb2.Metric.MetricType.COUNTER_METRIC
            
            self._inc_counter(self.call_count, value, timestamp_us)

    def inc_exception_count(self, value, timestamp_us):
        with self._update_lock:
            if not self.exception_count:
                self.exception_count = signals_pb2.Metric()
                self.exception_count.type = signals_pb2.Metric.MetricType.COUNTER_METRIC
            
            self._inc_counter(self.exception_count, value, timestamp_us)

    def inc_data_counter(self, data_name, counter_name, value, timestamp_us):
        with self._update_lock:
            key = (data_name, counter_name)
            if key not in self.data_counters:
                counter = signals_pb2.DataMetric()
                counter.data_name = data_name
                counter.metric_name = counter_name
                counter.metric.type = signals_pb2.Metric.MetricType.COUNTER_METRIC
                self.data_counters[key] = counter
            else:
                counter = self.data_counters[key]

            self._inc_counter(counter.metric, value, timestamp_us)

    def finalize(self, timestamp_us):
        with self._update_lock:
            if self.call_count:
                self._finalize_counter(self.call_count, timestamp_us)
            if self.exception_count:
                self._finalize_counter(self.exception_count, timestamp_us)
            for counter in self.data_counters.values():
                self._finalize_counter(counter.metric, timestamp_us)

    def _inc_counter(self, counter, value, timestamp_us):
        bucket = int(timestamp_us / 1e6)
        if bucket in counter.counter.buckets:
            counter.counter.buckets[bucket] += value
            counter.counter.buckets[self._start_sec] = 0
        else:
            counter.counter.buckets[self._start_sec] = 0
            counter.counter.buckets[bucket] = value

    def _finalize_counter(self, counter, timestamp_us):
        end_sec = int(timestamp_us / 1e6)
        if end_sec not in counter.counter.buckets:
            counter.counter.buckets[end_sec] = 0


def _check_module(name):
    return name in sys.modules


def _sha1(text, size=-1):
    sha1_hash = hashlib.sha1()
    sha1_hash.update(text.encode('utf-8'))
    return sha1_hash.hexdigest()[0:size]


def _uuid_sha1(size=-1):
    return _sha1(str(uuid.uuid4()), size)
