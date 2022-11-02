from typing import Optional
import logging

from tensorflow import keras
from tensorflow.keras.callbacks import Callback

import graphsignal
from graphsignal.proto import signals_pb2
from graphsignal.proto_utils import parse_semver

logger = logging.getLogger('graphsignal')


class GraphsignalCallback(Callback):
    def __init__(self, tags: Optional[dict] = None):
        super().__init__()

        from graphsignal.profilers.tensorflow import TensorFlowProfiler
        self._profiler = TensorFlowProfiler()
        self._keras_version = None
        self._trace = None
        self._tags = tags

    def on_train_begin(self, logs=None):
        self._configure_profiler()

    def on_train_end(self, logs=None):
        graphsignal.upload()

    def on_train_batch_begin(self, batch, logs=None):
        self._start_trace('train_batch', batch)

    def on_train_batch_end(self, batch, logs=None):
        self._stop_trace()

    def on_test_begin(self, logs=None):
        self._configure_profiler()

    def on_test_end(self, logs=None):
        graphsignal.upload()

    def on_test_batch_begin(self, batch, logs=None):
        self._start_trace('test_batch', batch)

    def on_test_batch_end(self, batch, logs=None):
        self._stop_trace()

    def on_predict_begin(self, logs=None):
        self._configure_profiler()

    def on_predict_end(self, logs=None):
        graphsignal.upload()

    def on_predict_batch_begin(self, batch, logs=None):
        self._start_trace('predict_batch', batch)

    def on_predict_batch_end(self, batch, logs=None):
        self._stop_trace()

    def _configure_profiler(self):
        try:
            self._keras_version = signals_pb2.SemVer()
            parse_semver(self._keras_version, keras.__version__)
        except Exception:
            logger.error('Error configuring Keras profiler', exc_info=True)

    def _start_trace(self, endpoint, batch_idx):
        if not self._trace:
            self._trace = graphsignal.start_trace(
                endpoint=endpoint,
                tags=self._tags,
                profiler=self._profiler)
            self._trace.set_tag('batch', batch_idx)

    def _stop_trace(self):
        if self._trace:
            if self._trace.is_tracing():
                self._update_signal()
            self._trace.stop()
            self._trace = None

    def _update_signal(self):
        try:
            signal = self._trace._signal

            signal.agent_info.framework_profiler_type = signals_pb2.AgentInfo.ProfilerType.KERAS_PROFILER

            framework = signal.frameworks.add()
            framework.type = signals_pb2.FrameworkInfo.FrameworkType.KERAS_FRAMEWORK
            framework.version.CopyFrom(self._keras_version)
        except Exception as exc:
            logger.debug('Error in Hugging Face callback', exc_info=True)
            self._trace._add_profiler_exception(exc)
