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
    def __init__(self, endpoint: str, tags: Optional[dict] = None):
        super().__init__()
        self._pl_version = None
        self._tracer = graphsignal.tracer(with_profiler='pytorch')
        self._span = None
        self._endpoint = endpoint
        self._tags = tags
        self._model_size_mb = None

    def on_validate_start(self, trainer, pl_module):
        self._configure_profiler(trainer, pl_module)

    def on_validate_end(self, trainer, pl_module):
        graphsignal.upload()

    def on_validate_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        self._start_profiler(trainer)

    def on_validate_batch_end(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        self._stop_profiler(trainer)

    def on_test_start(self, trainer, pl_module):
        self._configure_profiler(trainer, pl_module)

    def on_test_end(self, trainer, pl_module):
        graphsignal.upload()

    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        self._start_profiler(trainer)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self._stop_profiler(trainer)

    def on_predict_start(self, trainer, pl_module):
        self._configure_profiler(trainer, pl_module)

    def on_predict_end(self, trainer, pl_module):
        graphsignal.upload()

    def on_predict_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        self._start_profiler(trainer)

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self._stop_profiler(trainer)

    def _configure_profiler(self, trainer, pl_module):
        try:
            self._pl_version = signals_pb2.SemVer()
            parse_semver(self._pl_version, pytorch_lightning.__version__)

            model_size_mb = pytorch_lightning.utilities.memory.get_model_size_mb(pl_module)
            if model_size_mb:
                self._model_size_mb = model_size_mb
        except Exception:
            logger.error('Error configuring PyTorch Lightning profiler', exc_info=True)

    def _start_profiler(self, trainer):
        if not self._span:
            self._span = self._tracer.span(
                endpoint=self._endpoint,
                tags=self._tags)

    def _stop_profiler(self, trainer):
        if self._span:
            if self._span.is_tracing():
                self._update_profile(trainer)
            self._span.stop()
            self._span = None

    def _update_profile(self, trainer):
        try:
            signal = self._span._signal

            signal.agent_info.framework_profiler_type = signals_pb2.AgentInfo.ProfilerType.PYTORCH_LIGHTNING_PROFILER

            framework = signal.frameworks.add()
            framework.type = signals_pb2.FrameworkInfo.FrameworkType.PYTORCH_LIGHTNING_FRAMEWORK
            framework.version.CopyFrom(self._pl_version)

            signal.model_info.model_format = signals_pb2.ModelInfo.ModelFormat.PYTORCH_FORMAT
            if self._model_size_mb:
                signal.model_info.model_size_bytes = int(self._model_size_mb * 1e6)
        except Exception as exc:
            self._span._add_profiler_exception(exc)
