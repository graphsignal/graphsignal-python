import logging
import sys
import threading
import random
import time
import uuid
import hashlib
import importlib
import socket
import contextvars

import graphsignal
from graphsignal import version
from graphsignal.uploader import Uploader
from graphsignal.metrics import MetricStore
from graphsignal.trace_sampler import TraceSampler
from graphsignal.proto import signals_pb2
from graphsignal.detectors.latency_outlier_detector import LatencyOutlierDetector
from graphsignal.detectors.missing_value_detector import MissingValueDetector
from graphsignal.proto_utils import parse_semver

logger = logging.getLogger('graphsignal')


class Agent:
    METRIC_READ_INTERVAL_SEC = 10
    METRIC_UPLOAD_INTERVAL_SEC = 20

    def __init__(
            self, 
            api_key=None, 
            api_url=None, 
            deployment=None, 
            tags=None, 
            auto_instrument=True, 
            upload_on_shutdown=True, 
            debug_mode=False):
        self.api_key = api_key
        if api_url:
            self.api_url = api_url
        else:
            self.api_url = 'https://agent-api.graphsignal.com'
        self.deployment = deployment
        self.tags = tags
        self.context_tags = None
        self.params = None
        self.auto_instrument = auto_instrument
        self.upload_on_shutdown = upload_on_shutdown
        self.debug_mode = debug_mode
        self.hostname = None
        try:
            self.hostname = socket.gethostname()
        except BaseException:
            logger.debug('Error reading hostname', exc_info=True)

        self._uploader = None
        self._trace_samplers = None
        self._metric_store = None
        self._recorders = None
        self._lo_detectors = None
        self._mv_detector = None

        self._process_start_ms = int(time.time() * 1e3)

        self.last_metric_read_ts = 0
        self.last_metric_upload_ts = int(self._process_start_ms / 1e3)

    def setup(self):
        self._uploader = Uploader()
        self._uploader.setup()
        self._trace_samplers = {}
        self._metric_store = MetricStore()
        self._recorders = {}
        self._lo_detectors = {}

        # initialize context tags variable
        self.context_tags = contextvars.ContextVar('graphsignal_context_tags', default={})

        # pre-initialize recorders to enable auto-instrumentation for packages imported before graphsignal.configure()
        # as a fallback, any other trace sample will try to initialize uninitialized supported recorders
        # in a worst case scenario, this will result in a first execution not being traced
        self.recorders()

    def shutdown(self):
        # Create snapshot signals to send final metrics.
        if self.upload_on_shutdown:
            if self._metric_store.has_unexported():
                metrics = self._metric_store.export()
                for metric in metrics:
                    self._uploader.upload_metric(metric)

        for recorder in self._recorders.values():
            recorder.shutdown()

        self.upload(block=True)

        self._recorders = None
        self._trace_samplers = None
        self._metric_store = None
        self._lo_detectors = None
        self._mv_detector = None
        self._uploader = None

        self.context_tags.set({})
        self.context_tags = None

    def uploader(self):
        return self._uploader

    def recorders(self):
        recorder_specs = [
            ('graphsignal.recorders.cprofile_recorder', 'CProfileRecorder', None, ['torch', 'yappi']),
            ('graphsignal.recorders.yappi_recorder', 'YappiRecorder', 'yappi', 'torch'),
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
            ('graphsignal.recorders.langchain_recorder', 'LangChainRecorder', 'langchain', None),
            ('graphsignal.recorders.banana_recorder', 'BananaRecorder', 'banana_dev', None),
        ]
        last_exc = None
        for module_name, class_name, include, exclude in recorder_specs:
            try:
                key = (module_name, class_name)

                if exclude:
                    exclude = [exclude] if isinstance(exclude, str) else exclude
                    is_excluded = False
                    for mod in exclude:
                        if _check_module(mod):
                            if key in self._recorders:
                                del self._recorders[key]
                            is_excluded = True
                            break
                    if is_excluded:
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

    def metric_store(self):
        return self._metric_store

    def lo_detector(self, endpoint):
        if endpoint in self._lo_detectors:
            return self._lo_detectors[endpoint]
        else:
            lo_detector = self._lo_detectors[endpoint] = LatencyOutlierDetector()
            return lo_detector

    def mv_detector(self):
        if self._mv_detector is None:
            self._mv_detector = MissingValueDetector()
        return self._mv_detector

    def emit_trace_start(self, proto, context, options):
        last_exc = None
        for recorder in reversed(self.recorders()):
            try:
                recorder.on_trace_start(proto, context, options)
            except Exception as exc:
                last_exc = exc
        if last_exc:
            raise last_exc

    def emit_trace_stop(self, proto, context, options):
        last_exc = None
        for recorder in self.recorders():
            try:
                recorder.on_trace_stop(proto, context, options)
            except Exception as exc:
                last_exc = exc
        if last_exc:
            raise last_exc

    def emit_trace_read(self, proto, context, options):
        last_exc = None
        for recorder in self.recorders():
            try:
                recorder.on_trace_read(proto, context, options)
            except Exception as exc:
                last_exc = exc
        if last_exc:
            raise last_exc

    def emit_metric_update(self):
        last_exc = None
        for recorder in self.recorders():
            try:
                recorder.on_metric_update()
            except Exception as exc:
                last_exc = exc
        if last_exc:
            raise last_exc

    def create_trace_proto(self):
        proto = signals_pb2.Trace()
        proto.trace_id = _uuid_sha1(size=12)
        proto.agent_info.agent_type = signals_pb2.AgentInfo.AgentType.PYTHON_AGENT
        parse_semver(proto.agent_info.version, version.__version__)
        return proto

    def upload(self, block=False):
        if block:
            self._uploader.flush()
        else:
            self._uploader.flush_in_thread()

    def check_metric_read_interval(self, now=None):
        if now is None:
            now = time.time()
        return (self.last_metric_read_ts < now - Agent.METRIC_READ_INTERVAL_SEC)
    
    def set_metric_read(self, now=None):
        self.last_metric_read_ts = now if now else time.time()

    def check_metric_upload_interval(self, now=None):
        if now is None:
            now = time.time()
        return (self.last_metric_upload_ts < now - Agent.METRIC_UPLOAD_INTERVAL_SEC)

    def set_metric_upload(self, now=None):
        self.last_metric_upload_ts = now if now else time.time()

    def tick(self, block=False, now=None):
        if now is None:
            now = time.time()
        if self.check_metric_upload_interval(now):
            if self._metric_store.has_unexported():
                metrics = self._metric_store.export()
                for metric in metrics:
                    self.uploader().upload_metric(metric)
                self.set_metric_upload(now)

        self.upload(block=False)


def _check_module(module_name):
    return module_name in sys.modules


def _sha1(text, size=-1):
    sha1_hash = hashlib.sha1()
    sha1_hash.update(text.encode('utf-8'))
    return sha1_hash.hexdigest()[0:size]


def _uuid_sha1(size=-1):
    return _sha1(str(uuid.uuid4()), size)
