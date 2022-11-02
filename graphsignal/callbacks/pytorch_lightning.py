from typing import Optional
import logging
import time
from torch import Tensor
import pytorch_lightning
from pytorch_lightning.callbacks.base import Callback

import graphsignal
from graphsignal.proto import signals_pb2
from graphsignal.proto_utils import parse_semver

logger = logging.getLogger('graphsignal')


class GraphsignalCallback(Callback):
    def __init__(self, tags: Optional[dict] = None):
        super().__init__()

        from graphsignal.profilers.pytorch import PyTorchProfiler
        self._profiler = PyTorchProfiler()
        self._pl_version = None
        self._trace = None
        self._tags = tags
        self._model_size_mb = None

    def on_train_start(self, trainer, pl_module):
        self._configure_profiler(trainer, pl_module)

    def on_train_end(self, trainer, pl_module):
        graphsignal.upload()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self._start_trace('train_batch', batch_idx)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._stop_trace()

    def on_validation_start(self, trainer, pl_module):
        self._configure_profiler(trainer, pl_module)

    def on_validation_end(self, trainer, pl_module):
        graphsignal.upload()

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        self._start_trace('validate_batch', batch_idx)

    def on_validation_batch_end(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        self._stop_trace()

    def on_test_start(self, trainer, pl_module):
        self._configure_profiler(trainer, pl_module)

    def on_test_end(self, trainer, pl_module):
        graphsignal.upload()

    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        self._start_trace('test_batch', batch_idx)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self._stop_trace()

    def on_predict_start(self, trainer, pl_module):
        self._configure_profiler(trainer, pl_module)

    def on_predict_end(self, trainer, pl_module):
        graphsignal.upload()

    def on_predict_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        self._start_trace('predict_batch', batch_idx)

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self._stop_trace()

    def _configure_profiler(self, trainer, pl_module):
        try:
            self._pl_version = signals_pb2.SemVer()
            parse_semver(self._pl_version, pytorch_lightning.__version__)

            model_size_mb = pytorch_lightning.utilities.memory.get_model_size_mb(pl_module)
            if model_size_mb:
                self._model_size_mb = model_size_mb
        except Exception:
            logger.error('Error configuring PyTorch Lightning profiler', exc_info=True)

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

            signal.agent_info.framework_profiler_type = signals_pb2.AgentInfo.ProfilerType.PYTORCH_LIGHTNING_PROFILER

            framework = signal.frameworks.add()
            framework.type = signals_pb2.FrameworkInfo.FrameworkType.PYTORCH_LIGHTNING_FRAMEWORK
            framework.version.CopyFrom(self._pl_version)

            signal.model_info.model_format = signals_pb2.ModelInfo.ModelFormat.PYTORCH_FORMAT
            if self._model_size_mb:
                signal.model_info.model_size_bytes = int(self._model_size_mb * 1e6)
        except Exception as exc:
            logger.debug('Error in Hugging Face callback', exc_info=True)
            self._trace._add_profiler_exception(exc)
