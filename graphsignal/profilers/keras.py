from typing import Optional
import logging

from tensorflow import keras
from tensorflow.keras.callbacks import Callback

import graphsignal
from graphsignal.proto import profiles_pb2
from graphsignal.proto_utils import parse_semver
from graphsignal.profilers.tensorflow import TensorflowProfiler
from graphsignal.inference_span import InferenceSpan

logger = logging.getLogger('graphsignal')


class GraphsignalCallback(Callback):
    def __init__(self, batch_size: Optional[int] = None):
        super().__init__()
        self._keras_version = None
        self._profiler = TensorflowProfiler()
        self._span = None
        self._batch_size = batch_size

    def on_test_begin(self, logs=None):
        self._configure_profiler()

    def on_test_end(self, logs=None):
        graphsignal.upload()

    def on_test_batch_begin(self, batch, logs=None):
        self._start_profiler()

    def on_test_batch_end(self, batch, logs=None):
        self._stop_profiler()
        self._log_metrics(logs)

    def on_predict_begin(self, logs=None):
        self._configure_profiler()

    def on_predict_end(self, logs=None):
        graphsignal.upload()

    def on_predict_batch_begin(self, batch, logs=None):
        self._start_profiler()

    def on_predict_batch_end(self, batch, logs=None):
        self._stop_profiler()
        self._log_metrics(logs)

    def _configure_profiler(self):
        try:
            self._keras_version = profiles_pb2.SemVer()
            parse_semver(self._keras_version, keras.__version__)

            if self._batch_size:
                graphsignal.log_parameter('batch_size', self._batch_size)
        except Exception:
            logger.error('Error configuring Keras profiler', exc_info=True)

    def _start_profiler(self):
        if not self._span:
            self._span = InferenceSpan(
                batch_size=self._batch_size,
                operation_profiler=self._profiler)

    def _stop_profiler(self):
        if self._span:
            if self._span._is_scheduled:
                self._update_profile()
            self._span.stop()
            self._span = None

    def _update_profile(self):
        try:
            profile = self._span._profile

            profile.profiler_info.framework_profiler_type = profiles_pb2.ProfilerInfo.ProfilerType.KERAS_PROFILER

            framework = profile.frameworks.add()
            framework.type = profiles_pb2.FrameworkInfo.FrameworkType.KERAS_FRAMEWORK
            framework.version.CopyFrom(self._keras_version)
        except Exception as exc:
            self._span._add_profiler_exception(exc)

    def _log_metrics(self, logs):
        if logs:
            for key, value in logs.items():
                if isinstance(key, str) and isinstance(value, (int, float)):
                    graphsignal.log_metric(key, value)
