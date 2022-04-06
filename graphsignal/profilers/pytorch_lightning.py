import logging

from pytorch_lightning.callbacks.base import Callback

from graphsignal.proto import profiles_pb2
from graphsignal.profilers.pytorch import PyTorchProfiler
from graphsignal.profiling_span import ProfilingSpan

logger = logging.getLogger('graphsignal')


class GraphsignalCallback(Callback):
    __slots__ = [
        '_profiler',
        '_span'
    ]

    def __init__(self):
        self._profiler = PyTorchProfiler()
        self._span = None
        self._action_name = None
        super().__init__()

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

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self._start_profiler(
            'Training batch', profiles_pb2.Span.SpanType.TRAINING_BATCH, batch=batch_idx)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._stop_profiler()

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        self._start_profiler(
            'Validation batch', profiles_pb2.Span.SpanType.VALIDATION_BATCH, batch=batch_idx)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self._stop_profiler()

    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        self._start_profiler(
            'Test batch', profiles_pb2.Span.SpanType.TEST_BATCH, batch=batch_idx)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self._stop_profiler()
