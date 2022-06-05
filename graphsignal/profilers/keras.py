from typing import Optional
import logging

from tensorflow.keras.callbacks import Callback

import graphsignal
from graphsignal.proto import profiles_pb2
from graphsignal.profilers.tensorflow import TensorflowProfiler
from graphsignal.profiling_step import ProfilingStep

logger = logging.getLogger('graphsignal')

PHASE_TRAINING = 'training'
PHASE_TEST = 'test'
PHASE_PREDICTION = 'prediction'


class GraphsignalCallback(Callback):
    def __init__(self, batch_size: Optional[int] = None):
        super().__init__()
        self._profiler = TensorflowProfiler()
        self._step = None
        self._batch_size = batch_size

    def on_train_begin(self, logs=None):
        self._configure_profiler()

    def on_train_end(self, logs=None):
        self._stop_profiler()

    def on_test_begin(self, logs=None):
        self._configure_profiler()

    def on_test_end(self, logs=None):
        self._stop_profiler()

    def on_predict_begin(self, logs=None):
        self._configure_profiler()

    def on_predict_end(self, logs=None):
        self._stop_profiler()

    def on_train_batch_begin(self, batch, logs=None):
        self._stop_profiler()
        self._start_profiler(PHASE_TRAINING)

    def on_train_batch_end(self, batch, logs=None):
        self._log_metrics(logs)

    def on_test_batch_begin(self, batch, logs=None):
        self._stop_profiler()
        self._start_profiler(PHASE_TEST)

    def on_test_batch_end(self, batch, logs=None):
        self._log_metrics(logs)

    def on_predict_batch_begin(self, batch, logs=None):
        self._stop_profiler()
        self._start_profiler(PHASE_PREDICTION)

    def on_predict_batch_end(self, batch, logs=None):
        self._log_metrics(logs)

    def _configure_profiler(self):
        if self._batch_size:
            graphsignal.log_parameter('batch_size', self._batch_size)

    def _start_profiler(self, phase_name):
        if not self._step:
            self._step = ProfilingStep(
                phase_name=phase_name,
                effective_batch_size=self._batch_size,
                framework_profiler=self._profiler)

    def _stop_profiler(self):
        if self._step:
            if self._step._is_scheduled:
                self._update_profile()
            self._step.stop()
            self._step = None

    def _update_profile(self):
        profile = self._step._profile

        if self._batch_size:
            profile.step_stats.batch_size = self._batch_size

    def _log_metrics(self, logs):
        if logs:
            for key, value in logs.items():
                if isinstance(key, str) and isinstance(value, (int, float)):
                    graphsignal.log_metric(key, value)
