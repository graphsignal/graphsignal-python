import logging

from pytorch_lightning.callbacks.base import Callback

import graphsignal
from graphsignal.profilers.pytorch import profile_span

logger = logging.getLogger('graphsignal')


class GraphsignalCallback(Callback):
    __slots__ = [
        '_span'
    ]

    def __init__(self):
        self._span = None
        self._action_name = None
        super().__init__()

    def _start_profiler(self, span_name, batch=None):
        if not self._span:
            self._span = profile_span(span_name=span_name)
            if batch is not None:
                self._span.add_metadata('batch', batch)

    def _stop_profiler(self):
        if self._span:
            self._span.stop()
            self._span = None

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self._start_profiler('Training batch', batch=batch_idx)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._stop_profiler()

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        self._start_profiler('Validation batch', batch=batch_idx)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self._stop_profiler()

    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        self._start_profiler('Test batch', batch=batch_idx)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self._stop_profiler()
