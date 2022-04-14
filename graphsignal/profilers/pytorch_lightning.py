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

    def _start_profiler(self, run_phase):
        if not self._span:
            self._span = ProfilingSpan(
                run_phase=run_phase,
                framework_profiler=self._profiler)

    def _stop_profiler(self):
        if self._span:
            self._span.stop()
            self._span = None

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self._start_profiler(profiles_pb2.RunPhase.TRAINING)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._stop_profiler()

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        self._start_profiler(profiles_pb2.RunPhase.VALIDATION)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self._stop_profiler()

    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        self._start_profiler(profiles_pb2.RunPhase.TEST)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self._stop_profiler()
