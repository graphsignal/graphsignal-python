import logging
import threading
import random
import time
import uuid
import hashlib

import graphsignal
from graphsignal.tracer import Tracer
from graphsignal.uploader import Uploader
from graphsignal.usage.process_reader import ProcessReader
from graphsignal.usage.nvml_reader import NvmlReader
from graphsignal.trace_sampler import TraceSampler
from graphsignal.proto import signals_pb2

logger = logging.getLogger('graphsignal')


class Agent:
    def __init__(self, api_key, debug_mode=False):
        self.worker_id = _uuid_sha1(size=12)
        self.api_key = api_key
        self.debug_mode = debug_mode

        self._uploader = None
        self._process_reader = None
        self._nvml_reader = None
        self._trace_samplers = None
        self._span_stats = None
        self._tracers = None

    def start(self):
        self._uploader = Uploader()
        self._uploader.setup()
        self._process_reader = ProcessReader()
        self._process_reader.setup()
        self._nvml_reader = NvmlReader()
        self._nvml_reader.setup()
        self._trace_samplers = {}
        self._span_stats = {}
        self._tracers = {}

    def stop(self):
        self.upload(block=True)
        self._process_reader.shutdown()
        self._nvml_reader.shutdown()
        self._trace_samplers = None
        self._span_stats = None
        self._tracers = None

    def uploader(self):
        return self._uploader

    def tracer(self, with_profiler=True):
        if with_profiler in self._tracers:
            return self._tracers[with_profiler]

        profiler = None
        if with_profiler == True or with_profiler == 'python':
            from graphsignal.profilers.python import PythonProfiler
            profiler = PythonProfiler()
        elif with_profiler == 'tensorflow':
            from graphsignal.profilers.tensorflow import TensorFlowProfiler
            profiler = TensorFlowProfiler()
        elif with_profiler == 'pytorch':
            from graphsignal.profilers.pytorch import PyTorchProfiler
            profiler = PyTorchProfiler()
        elif with_profiler == 'jax':
            from graphsignal.profilers.jax import JaxProfiler
            profiler = JaxProfiler()
        elif with_profiler == 'onnxruntime':
            from graphsignal.profilers.onnxruntime import ONNXRuntimeProfiler
            profiler = ONNXRuntimeProfiler()
        elif with_profiler:
            raise ValueError('Invalid profiler name: {0}'.format(with_profiler))

        tracer = self._tracers[with_profiler] = Tracer(profiler=profiler)
        return tracer

    def get_trace_sampler(self, model_name):
        if model_name in self._trace_samplers:
            return self._trace_samplers[model_name]
        else:
            trace_sampler = self._trace_samplers[model_name] = TraceSampler()
            return trace_sampler

    def reset_span_stats(self, model_name):
        self._span_stats[model_name] = SpanStats()

    def get_span_stats(self, model_name):
        if model_name not in self._span_stats:
            self.reset_span_stats(model_name)

        return self._span_stats[model_name]

    def read_usage(self, signal):
        self._process_reader.read(signal)
        self._nvml_reader.read(signal)

    def create_signal(self):
        signal = signals_pb2.MLSignal()
        signal.worker_id = graphsignal._agent.worker_id
        signal.signal_id = _uuid_sha1(size=12)
        return signal

    def upload(self, block=False):
        if block:
            self._uploader.flush()
        else:
            self._uploader.flush_in_thread()

    def tick(self, block=False):
        self.upload(block=False)


class SpanStats:
    MAX_RESERVOIR_SIZE = 100
    MAX_COUNTERS = 10

    def __init__(self):
        self._update_lock = threading.Lock()
        self._start_sec = int(time.time())
        self.time_reservoir_us = []
        self.call_counter = signals_pb2.SpanStats.Counter()
        self.exception_counter = signals_pb2.SpanStats.Counter()
        self.data_counters = {}

    def add_time(self, duration_us):
        with self._update_lock:
            if len(self.time_reservoir_us) < SpanStats.MAX_RESERVOIR_SIZE:
                self.time_reservoir_us.append(duration_us)
            else:
                self.time_reservoir_us[random.randint(0, SpanStats.MAX_RESERVOIR_SIZE - 1)] = duration_us

    def inc_call_counter(self, value, timestamp_us):
        with self._update_lock:
            self._inc_counter(self.call_counter, value, timestamp_us)

    def inc_exception_counter(self, value, timestamp_us):
        with self._update_lock:
            self._inc_counter(self.exception_counter, value, timestamp_us)

    def inc_data_counter(self, name, value, unit, timestamp_us):
        with self._update_lock:
            if name not in self.data_counters:
                if len(self.data_counters) < SpanStats.MAX_COUNTERS:
                    counter = self.data_counters[name] = signals_pb2.SpanStats.Counter()
                    counter.unit = unit
                else:
                    return
            else:
                counter = self.data_counters[name]

            self._inc_counter(counter, value, timestamp_us)

    def finalize(self, timestamp_us):
        with self._update_lock:
            self._finalize_counter(self.call_counter, timestamp_us)
            self._finalize_counter(self.exception_counter, timestamp_us)
            for data_counter in self.data_counters.values():
                self._finalize_counter(data_counter, timestamp_us)

    def _inc_counter(self, counter, value, timestamp_us):
        bucket = int(timestamp_us / 1e6)
        if bucket in counter.buckets_sec:
            counter.buckets_sec[bucket] += value
            counter.buckets_sec[self._start_sec] = 0
        else:
            counter.buckets_sec[self._start_sec] = 0
            counter.buckets_sec[bucket] = value

    def _finalize_counter(self, counter, timestamp_us):
        end_sec = int(timestamp_us / 1e6)
        if end_sec not in counter.buckets_sec:
            counter.buckets_sec[end_sec] = 0


def _sha1(text, size=-1):
    sha1_hash = hashlib.sha1()
    sha1_hash.update(text.encode('utf-8'))
    return sha1_hash.hexdigest()[0:size]


def _uuid_sha1(size=-1):
    return _sha1(str(uuid.uuid4()), size)
