import logging
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

logger = logging.getLogger('graphsignal')


class Agent:
    def __init__(self, api_key, disable_profiling=False, debug_mode=False):
        self.worker_id = _uuid_sha1(size=12)
        self.api_key = api_key
        self.disable_profiling = disable_profiling
        self.debug_mode = debug_mode
        self.trace_samplers = None
        self.inference_stats = None

    def start(self):
        self.uploader = Uploader()
        self.uploader.setup()
        self.process_reader = ProcessReader()
        self.process_reader.setup()
        self.nvml_reader = NvmlReader()
        self.nvml_reader.setup()
        self.trace_samplers = {}
        self.inference_stats = {}

    def stop(self):
        self.upload(block=True)
        self.process_reader.shutdown()
        self.nvml_reader.shutdown()
        self.trace_samplers = None
        self.inference_stats = None

    def get_trace_sampler(self, model_name):
        if model_name in self.trace_samplers:
            return self.trace_samplers[model_name]
        else:
            trace_sampler = self.trace_samplers[model_name] = TraceSampler()
            return trace_sampler

    def reset_inference_stats(self, name):
        self.inference_stats[name] = InferenceStats()

    def get_inference_stats(self, name):
        if name not in self.inference_stats:
            self.reset_inference_stats(name)

        return self.inference_stats[name]

    def create_signal(self):
        signal = signals_pb2.MLSignal()
        signal.worker_id = graphsignal._agent.worker_id
        signal.signal_id = _uuid_sha1(size=12)
        return signal

    def upload(self, block=False):
        if block:
            graphsignal._agent.uploader.flush()
        else:
            graphsignal._agent.uploader.flush_in_thread()

    def tick(self, block=False):
        self.upload(block=False)


class InferenceStats:
    MAX_RESERVOIR_SIZE = 100
    MAX_COUNTERS = 10

    def __init__(self):
        self._update_lock = threading.Lock()
        self._start_sec = int(time.time())
        self.time_reservoir_us = []
        self.inference_counter = {}
        self.exception_counter = {}
        self.extra_counters = {}

    def add_time(self, duration_us):
        with self._update_lock:
            if len(self.time_reservoir_us) < InferenceStats.MAX_RESERVOIR_SIZE:
                self.time_reservoir_us.append(duration_us)
            else:
                self.time_reservoir_us[random.randint(0, InferenceStats.MAX_RESERVOIR_SIZE - 1)] = duration_us

    def inc_inference_counter(self, value, timestamp_us):
        with self._update_lock:
            self._inc_counter(self.inference_counter, value, timestamp_us)

    def inc_exception_counter(self, value, timestamp_us):
        with self._update_lock:
            self._inc_counter(self.exception_counter, value, timestamp_us)

    def inc_extra_counter(self, name, value, timestamp_us):
        with self._update_lock:
            if name not in self.extra_counters:
                if len(self.extra_counters) < InferenceStats.MAX_COUNTERS:
                    counter = self.extra_counters[name] = {}
                else:
                    return
            else:
                counter = self.extra_counters[name]

            self._inc_counter(counter, value, timestamp_us)

    def finalize(self, timestamp_us):
        with self._update_lock:
            self._finalize_counter(self.inference_counter, timestamp_us)
            self._finalize_counter(self.exception_counter, timestamp_us)
            for extra_counter in self.extra_counters.values():
                self._finalize_counter(extra_counter, timestamp_us)

    def _inc_counter(self, counter, value, timestamp_us):
        bucket = int(timestamp_us / 1e6)
        if bucket in counter:
            counter[bucket] += value
            counter[self._start_sec] = 0
        else:
            counter[self._start_sec] = 0
            counter[bucket] = value

    def _finalize_counter(self, counter, timestamp_us):
        end_sec = int(timestamp_us / 1e6)
        if end_sec not in counter:
            counter[end_sec] = 0


def _sha1(text, size=-1):
    sha1_hash = hashlib.sha1()
    sha1_hash.update(text.encode('utf-8'))
    return sha1_hash.hexdigest()[0:size]


def _uuid_sha1(size=-1):
    return _sha1(str(uuid.uuid4()), size)
