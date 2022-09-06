from typing import Optional
import logging

from tensorflow import keras
from tensorflow.keras.callbacks import Callback

import graphsignal
from graphsignal.proto import signals_pb2
from graphsignal.proto_utils import parse_semver

logger = logging.getLogger('graphsignal')


class GraphsignalCallback(Callback):
    def __init__(self, model_name: str, tags: Optional[dict] = None, batch_size: Optional[int] = None):
        super().__init__()
        self._keras_version = None
        self._tracer = graphsignal.tracer(with_profiler='tensorflow')
        self._span = None
        self._model_name = model_name
        self._tags = tags
        self._batch_size = batch_size

    def on_test_begin(self, logs=None):
        self._configure_profiler()

    def on_test_end(self, logs=None):
        graphsignal.upload()

    def on_test_batch_begin(self, batch, logs=None):
        self._start_profiler()

    def on_test_batch_end(self, batch, logs=None):
        self._stop_profiler()

    def on_predict_begin(self, logs=None):
        self._configure_profiler()

    def on_predict_end(self, logs=None):
        graphsignal.upload()

    def on_predict_batch_begin(self, batch, logs=None):
        self._start_profiler()

    def on_predict_batch_end(self, batch, logs=None):
        self._stop_profiler()

    def _configure_profiler(self):
        try:
            self._keras_version = signals_pb2.SemVer()
            parse_semver(self._keras_version, keras.__version__)
        except Exception:
            logger.error('Error configuring Keras profiler', exc_info=True)

    def _start_profiler(self):
        if not self._span:
            self._span = self._tracer.inference_span(
                model_name=self._model_name,
                tags=self._tags)
            self._span.measure_data(counts=dict(items=self._batch_size))

    def _stop_profiler(self):
        if self._span:
            if self._span.is_tracing():
                self._update_profile()
            self._span.stop()
            self._span = None

    def _update_profile(self):
        try:
            signal = self._span._signal

            signal.agent_info.framework_profiler_type = signals_pb2.AgentInfo.ProfilerType.KERAS_PROFILER

            framework = signal.frameworks.add()
            framework.type = signals_pb2.FrameworkInfo.FrameworkType.KERAS_FRAMEWORK
            framework.version.CopyFrom(self._keras_version)
        except Exception as exc:
            self._span._add_profiler_exception(exc)
