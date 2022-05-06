import logging

from tensorflow.keras.callbacks import Callback

from graphsignal.proto import profiles_pb2
from graphsignal.profilers.tensorflow import TensorflowProfiler
from graphsignal.profiling_step import ProfilingStep
from graphsignal import step_counter

logger = logging.getLogger('graphsignal')


class GraphsignalCallback(Callback):
    __slots__ = [
        '_profiler',
        '_step'
    ]

    def __init__(self):
        self._profiler = TensorflowProfiler()
        self._step = None

    def on_train_begin(self, logs=None):
        step_counter.init_step_stats(profiles_pb2.RunPhase.TRAINING)

    def on_train_end(self, logs=None):
        step_counter.reset_step_stats(profiles_pb2.RunPhase.TRAINING)

    def on_test_begin(self, logs=None):
        step_counter.init_step_stats(profiles_pb2.RunPhase.TEST)

    def on_test_end(self, logs=None):
        step_counter.reset_step_stats(profiles_pb2.RunPhase.TEST)

    def on_predict_begin(self, logs=None):
        step_counter.init_step_stats(profiles_pb2.RunPhase.PREDICTION)

    def on_predict_end(self, logs=None):
        step_counter.reset_step_stats(profiles_pb2.RunPhase.PREDICTION)

    def on_train_batch_begin(self, batch, logs=None):
        self._start_profiler(profiles_pb2.RunPhase.TRAINING)

    def on_train_batch_end(self, batch, logs=None):
        self._stop_profiler()

    def on_test_batch_begin(self, batch, logs=None):
        self._start_profiler(profiles_pb2.RunPhase.TEST)

    def on_test_batch_end(self, batch, logs=None):
        self._stop_profiler()

    def on_predict_batch_begin(self, batch, logs=None):
        self._start_profiler(profiles_pb2.RunPhase.VALIDATION)

    def on_predict_batch_end(self, batch, logs=None):
        self._stop_profiler()

    def _start_profiler(self, run_phase):
        if not self._step:
            self._step = ProfilingStep(
                run_phase=run_phase,
                framework_profiler=self._profiler)

    def _stop_profiler(self):
        if self._step:
            self._step.stop()
            self._step = None