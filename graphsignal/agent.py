import logging
import sys
import threading
import random
import time
import uuid
import hashlib
from collections import OrderedDict

import graphsignal
from graphsignal import version
from graphsignal.uploader import Uploader
from graphsignal.trace_sampler import TraceSampler
from graphsignal.proto import signals_pb2
from graphsignal.data.missing_value_detector import MissingValueDetector
from graphsignal.proto_utils import parse_semver

logger = logging.getLogger('graphsignal')


class Agent:
    def __init__(self, api_key=None, api_url=None, deployment=None, tags=None, debug_mode=False):
        self.worker_id = _uuid_sha1(size=12)
        self.api_key = api_key
        if api_url:
            self.api_url = api_url
        else:
            self.api_url = 'https://agent-api.graphsignal.com'
        self.deployment = deployment
        self.tags = tags
        self.params = None
        self.debug_mode = debug_mode

        self._uploader = None
        self._recorders = None
        self._trace_samplers = None
        self._metric_store = None
        self._mv_detector = None

        self._process_start_ms = int(time.time() * 1e3)

    def start(self):
        self._uploader = Uploader()
        self._uploader.setup()
        self._trace_samplers = {}
        self._metric_store = {}

    def stop(self):
        self.upload(block=True)
        self._recorders = None
        self._trace_samplers = None
        self._metric_store = None
        self._mv_detector = None

    def uploader(self):
        return self._uploader

    def recorders(self):
        if self._recorders is not None:
            return self._recorders
        
        supported_recorders = []

        if _check_module('torch'):
            from graphsignal.recorders.kineto_recorder import KinetoRecorder
            recorder = KinetoRecorder()
            supported_recorders.append(recorder)
        else:
            from graphsignal.recorders.cprofile_recorder import CProfileRecorder
            recorder = CProfileRecorder()
            supported_recorders.append(recorder)

        from graphsignal.recorders.process_recorder import ProcessRecorder
        recorder = ProcessRecorder()
        supported_recorders.append(recorder)

        from graphsignal.recorders.nvml_recorder import NVMLRecorder
        recorder = NVMLRecorder()
        supported_recorders.append(recorder)

        if _check_module('torch'):
            from graphsignal.recorders.pytorch_recorder import PyTorchRecorder
            recorder = PyTorchRecorder()
            supported_recorders.append(recorder)
        if _check_module('tensorflow'):
            from graphsignal.recorders.tensorflow_recorder import TensorFlowRecorder
            recorder = TensorFlowRecorder()
            supported_recorders.append(recorder)
        if _check_module('jax'):
            from graphsignal.recorders.jax_recorder import JAXRecorder
            recorder = JAXRecorder()
            supported_recorders.append(recorder)
        if _check_module('onnxruntime'):
            from graphsignal.recorders.onnxruntime_recorder import ONNXRuntimeRecorder
            recorder = ONNXRuntimeRecorder()
            supported_recorders.append(recorder)
        if _check_module('xgboost'):
            from graphsignal.recorders.xgboost_recorder import XGBoostRecorder
            recorder = XGBoostRecorder()
            supported_recorders.append(recorder)
        if _check_module('deepspeed'):
            from graphsignal.recorders.deepspeed_recorder import DeepSpeedRecorder
            recorder = DeepSpeedRecorder()
            supported_recorders.append(recorder)

        self._recorders = []
        last_exc = None
        for recorder in supported_recorders:
            try:
                recorder.setup()
                self._recorders.append(recorder)
            except Exception as exc:
                last_exc = exc
        if last_exc:
            raise last_exc

        return self._recorders

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
        if self._mv_detector is None:
            self._mv_detector = MissingValueDetector()
        return self._mv_detector

    def emit_trace_start(self, signal, context, options):
        last_exc = None
        for recorder in reversed(self.recorders()):
            try:
                recorder.on_trace_start(signal, context, options)
            except Exception as exc:
                last_exc = exc
        if last_exc:
            raise last_exc

    def emit_trace_stop(self, signal, context, options):
        last_exc = None
        for recorder in self.recorders():
            try:
                recorder.on_trace_stop(signal, context, options)
            except Exception as exc:
                last_exc = exc
        if last_exc:
            raise last_exc

    def emit_trace_read(self, signal, context, options):
        last_exc = None
        for recorder in self.recorders():
            try:
                recorder.on_trace_read(signal, context, options)
            except Exception as exc:
                last_exc = exc
        if last_exc:
            raise last_exc

    def create_signal(self):
        signal = signals_pb2.WorkerSignal()
        signal.worker_id = graphsignal._agent.worker_id
        signal.signal_id = _uuid_sha1(size=12)
        if self.deployment:
            signal.deployment_name = self.deployment
        signal.agent_info.agent_type = signals_pb2.AgentInfo.AgentType.PYTHON_AGENT
        parse_semver(signal.agent_info.version, version.__version__)
        return signal

    def upload(self, block=False):
        if block:
            self._uploader.flush()
        else:
            self._uploader.flush_in_thread()

    def tick(self, block=False):
        self.upload(block=False)


class MetricStore:
    MAX_RESERVOIR_SIZE = 250
    MAX_INTERVAL_SEC = 600

    def __init__(self):
        self._update_lock = threading.Lock()
        self.latency_us = None
        self.call_count = None
        self.exception_count = None
        self.data_counters = {}

    def add_time(self, duration_us):
        with self._update_lock:
            if not self.latency_us:
                self.latency_us = []
            if len(self.latency_us) < MetricStore.MAX_RESERVOIR_SIZE:
                self.latency_us.append(duration_us)
            else:
                self.latency_us[random.randint(0, MetricStore.MAX_RESERVOIR_SIZE - 1)] = duration_us

    def inc_call_count(self, value, timestamp_us):
        with self._update_lock:
            if not self.call_count:
                self.call_count = OrderedDict()
            self._inc_counter(self.call_count, value, timestamp_us)

    def inc_exception_count(self, value, timestamp_us):
        with self._update_lock:
            if not self.exception_count:
                self.exception_count = OrderedDict()
            self._inc_counter(self.exception_count, value, timestamp_us)

    def inc_data_counter(self, data_name, counter_name, value, timestamp_us):
        with self._update_lock:
            key = (data_name, counter_name)
            if key not in self.data_counters:
                counter = OrderedDict()
                self.data_counters[key] = counter
            else:
                counter = self.data_counters[key]
            self._inc_counter(counter, value, timestamp_us)

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
        if has_expired and start_bucket not in counter:
            counter[start_bucket] = 0

        counter[bucket] = counter.get(bucket, 0) + value

    def convert_to_proto(self, signal, timestamp_us):
        if self.latency_us and len(self.latency_us) > 0:
            self._convert_reservoir_to_proto(self.latency_us, signal.trace_metrics.latency_us, timestamp_us)
        if self.call_count and len(self.call_count) > 0:
            self._convert_counter_to_proto(self.call_count, signal.trace_metrics.call_count, timestamp_us)
        if self.exception_count and len(self.exception_count) > 0:
            self._convert_counter_to_proto(self.exception_count, signal.trace_metrics.exception_count, timestamp_us)
        for (data_name, counter_name), data_counter in self.data_counters.items():
            data_metric = signal.data_metrics.add()
            data_metric.data_name = data_name
            data_metric.metric_name = counter_name
            self._convert_counter_to_proto(data_counter, data_metric.metric, timestamp_us)

    def _convert_reservoir_to_proto(self, reservoir, metric, timestamp_us):
        metric.type = signals_pb2.Metric.MetricType.RESERVOIR_METRIC
        for value in reservoir:
            metric.reservoir.values.append(value)

    def _convert_counter_to_proto(self, counter, metric, timestamp_us):
        metric.type = signals_pb2.Metric.MetricType.GAUGE_METRIC
        interval_sec = int(timestamp_us / 1e6) - min(counter.keys()) + 1
        metric.gauge = sum(counter.values()) / interval_sec


def _check_module(name):
    return name in sys.modules


def _sha1(text, size=-1):
    sha1_hash = hashlib.sha1()
    sha1_hash.update(text.encode('utf-8'))
    return sha1_hash.hexdigest()[0:size]


def _uuid_sha1(size=-1):
    return _sha1(str(uuid.uuid4()), size)
