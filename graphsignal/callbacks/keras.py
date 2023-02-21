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

        self._is_initialized = False
        self._framework = None
        self._trace = None
        self._tags = tags

    def on_predict_begin(self, logs=None):
        self._configure()

    def on_predict_end(self, logs=None):
        graphsignal.upload()

    def on_predict_batch_begin(self, batch, logs=None):
        self._start_trace('predict_batch', batch)

    def on_predict_batch_end(self, batch, logs=None):
        self._stop_trace()

    def _configure(self):
        try:
            if not self._is_initialized:
                self._is_initialized = True

                self._framework = signals_pb2.FrameworkInfo()
                self._framework.name = 'Keras'
                parse_semver(self._framework.version, keras.__version__)
        except Exception:
            logger.error('Error configuring Keras callback', exc_info=True)

    def _start_trace(self, endpoint, batch_idx):
        if not self._trace:
            self._trace = graphsignal.start_trace(
                endpoint=endpoint,
                tags=self._tags)
            self._trace.set_tag('batch', batch_idx)

    def _stop_trace(self):
        if self._trace:
            if self._trace.is_sampling():
                self._update_signal()
            self._trace.stop()
            self._trace = None

    def _update_signal(self):
        try:
            signal = self._trace._signal

            if self._framework:
                signal.frameworks.append(self._framework)
        except Exception as exc:
            logger.debug('Error in Hugging Face callback', exc_info=True)
            self._trace._add_agent_exception(exc)
