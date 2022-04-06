import logging

from tensorflow.keras.callbacks import Callback

from graphsignal.proto import profiles_pb2
from graphsignal.profilers.tensorflow import TensorflowProfiler
from graphsignal.profiling_span import ProfilingSpan

logger = logging.getLogger('graphsignal')


class GraphsignalCallback(Callback):
    __slots__ = [
        '_profiler',
        '_span'
    ]

    def __init__(self):
        self._profiler = TensorflowProfiler()
        self._span = None

    def _start_profiler(self, span_name, span_type, batch=None):
        if not self._span:
            self._span = ProfilingSpan(
                framework_profiler=self._profiler,
                span_name=span_name,
                span_type=span_type)
            if batch is not None:
                self._span.add_metadata('batch', batch)

    def _stop_profiler(self):
        if self._span:
            self._span.stop()
            self._span = None

    def on_predict_begin(self, logs=None):
        self._start_profiler('Prediction')

    def on_predict_end(self, logs=None):
        self._stop_profiler()

    def on_train_batch_begin(self, batch, logs=None):
        self._start_profiler(
            'Training batch', profiles_pb2.Span.SpanType.TRAINING_BATCH, batch=batch)

    def on_train_batch_end(self, batch, logs=None):
        self._stop_profiler()

    def on_test_batch_begin(self, batch, logs=None):
        self._start_profiler(
            'Test batch', profiles_pb2.Span.SpanType.TEST_BATCH, batch=batch)

    def on_test_batch_end(self, batch, logs=None):
        self._stop_profiler()

    def on_predict_batch_begin(self, batch, logs=None):
        self._start_profiler(
            'Prediction batch', profiles_pb2.Span.SpanType.PREDICTION_BATCH, batch=batch)

    def on_predict_batch_end(self, batch, logs=None):
        self._stop_profiler()
