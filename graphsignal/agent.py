import logging
import sys
import threading
import random
import time
import uuid
import hashlib
import importlib

import graphsignal
from graphsignal import version
from graphsignal.uploader import Uploader
from graphsignal.metric_store import MetricStore
from graphsignal.trace_sampler import TraceSampler
from graphsignal.proto import signals_pb2
from graphsignal.data.missing_value_detector import MissingValueDetector
from graphsignal.proto_utils import parse_semver

logger = logging.getLogger('graphsignal')


class Agent:
    def __init__(self, api_key=None, api_url=None, deployment=None, tags=None, auto_instrument=True, debug_mode=False):
        self.worker_id = _uuid_sha1(size=12)
        self.api_key = api_key
        if api_url:
            self.api_url = api_url
        else:
            self.api_url = 'https://agent-api.graphsignal.com'
        self.deployment = deployment
        self.tags = tags
        self.params = None
        self.auto_instrument = auto_instrument
        self.debug_mode = debug_mode

        self._uploader = None
        self._trace_samplers = None
        self._metric_store = None
        self._mv_detector = None
        self._recorders = None

        self._process_start_ms = int(time.time() * 1e3)

    def setup(self):
        self._uploader = Uploader()
        self._uploader.setup()
        self._trace_samplers = {}
        self._metric_store = {}
        self._recorders = {}

        # pre-initialize recorders to enable auto-instrumentation for packages imported before graphsignal.configure()
        # as a fallback, any other trace sample will try to initialize uninitialized supported recorders
        # in a worst case scenario, this will result in a first execution not being traced
        self.recorders()

    def shutdown(self):
        for recorder in self._recorders.values():
            recorder.shutdown()
        self.upload(block=True)
        self._recorders = None
        self._trace_samplers = None
        self._metric_store = None
        self._mv_detector = None
        self._uploader = None

    def uploader(self):
        return self._uploader

    def recorders(self):
        recorder_specs = [
            ('graphsignal.recorders.cprofile_recorder', 'CProfileRecorder', None, 'torch'),
            ('graphsignal.recorders.kineto_recorder', 'KinetoRecorder', 'torch', None),
            ('graphsignal.recorders.process_recorder', 'ProcessRecorder', None, None),
            ('graphsignal.recorders.nvml_recorder', 'NVMLRecorder', None, None),
            ('graphsignal.recorders.pytorch_recorder', 'PyTorchRecorder', 'torch', None),
            ('graphsignal.recorders.tensorflow_recorder', 'TensorFlowRecorder', 'tensorflow', None),
            ('graphsignal.recorders.jax_recorder', 'JAXRecorder', 'jax', None),
            ('graphsignal.recorders.onnxruntime_recorder', 'ONNXRuntimeRecorder', 'onnxruntime', None),
            ('graphsignal.recorders.xgboost_recorder', 'XGBoostRecorder', 'xgboost', None),
            ('graphsignal.recorders.deepspeed_recorder', 'DeepSpeedRecorder', 'deepspeed', None),
            ('graphsignal.recorders.openai_recorder', 'OpenAIRecorder', 'openai', None),
        ]
        last_exc = None
        for module_name, class_name, include, exclude in recorder_specs:
            try:
                key = (module_name, class_name)

                if exclude and _check_module(exclude):
                    if key in self._recorders:
                        del self._recorders[key]
                    continue

                if key not in self._recorders:
                    if not include or _check_module(include):
                        module = importlib.import_module(module_name)
                        recorder_class = getattr(module, class_name)
                        recorder = recorder_class()
                        recorder.setup()
                        self._recorders[key] = recorder
            except Exception as exc:
                last_exc = exc
        if last_exc:
            raise last_exc

        return self._recorders.values()

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


def _check_module(module_name):
    return module_name in sys.modules


def _sha1(text, size=-1):
    sha1_hash = hashlib.sha1()
    sha1_hash.update(text.encode('utf-8'))
    return sha1_hash.hexdigest()[0:size]


def _uuid_sha1(size=-1):
    return _sha1(str(uuid.uuid4()), size)
