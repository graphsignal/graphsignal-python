from typing import Optional
import logging
import time
from torch import Tensor
import pytorch_lightning
from pytorch_lightning.callbacks.base import Callback

import graphsignal
from graphsignal.proto import profiles_pb2
from graphsignal.proto_utils import parse_semver
from graphsignal.profilers.pytorch import PyTorchProfiler
from graphsignal.inference_span import InferenceSpan

logger = logging.getLogger('graphsignal')


class GraphsignalCallback(Callback):
    def __init__(self, batch_size: Optional[int] = None):
        super().__init__()
        self._pl_version = None
        self._profiler = PyTorchProfiler()
        self._span = None
        self._batch_size = batch_size
        self._model_size_mb = None
        self._node_rank = -1
        self._local_rank = -1
        self._global_rank = -1

    def on_validate_start(self, trainer, pl_module):
        self._configure_profiler(trainer, pl_module)

    def on_validate_end(self, trainer, pl_module):
        graphsignal.upload()

    def on_validate_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        self._start_profiler(trainer)

    def on_validate_batch_end(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        self._stop_profiler(trainer)
        self._log_metrics(trainer.logged_metrics)

    def on_test_start(self, trainer, pl_module):
        self._configure_profiler(trainer, pl_module)

    def on_test_end(self, trainer, pl_module):
        graphsignal.upload()

    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        self._start_profiler(trainer)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self._stop_profiler(trainer)
        self._log_metrics(trainer.logged_metrics)

    def on_predict_start(self, trainer, pl_module):
        self._configure_profiler(trainer, pl_module)

    def on_predict_end(self, trainer, pl_module):
        graphsignal.upload()

    def on_predict_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        self._start_profiler(trainer)

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self._stop_profiler(trainer)
        self._log_metrics(trainer.logged_metrics)

    def _configure_profiler(self, trainer, pl_module):
        try:
            self._pl_version = profiles_pb2.SemVer()
            parse_semver(self._pl_version, pytorch_lightning.__version__)

            if self._check_param(trainer, 'node_rank') and trainer.node_rank >= 0:
                self._node_rank = trainer.node_rank
            if self._check_param(trainer, 'local_rank') and trainer.local_rank >= 0:
                self._local_rank = trainer.local_rank
            if self._check_param(trainer, 'global_rank') and trainer.global_rank >= 0:
                self._global_rank = trainer.global_rank

            model_size_mb = pytorch_lightning.utilities.memory.get_model_size_mb(pl_module)
            if model_size_mb:
                self._model_size_mb = model_size_mb
        except Exception:
            logger.error('Error configuring PyTorch Lightning profiler', exc_info=True)

    def _start_profiler(self, trainer):
        if not self._span:
            self._span = InferenceSpan(
                batch_size=self._batch_size,
                operation_profiler=self._profiler)

    def _stop_profiler(self, trainer):
        if self._span:
            if self._span._is_scheduled:
                self._update_profile(trainer)
            self._span.stop()
            self._span = None

    def _update_profile(self, trainer):
        try:
            profile = self._span._profile

            profile.profiler_info.framework_profiler_type = profiles_pb2.ProfilerInfo.ProfilerType.PYTORCH_LIGHTNING_PROFILER

            framework = profile.frameworks.add()
            framework.type = profiles_pb2.FrameworkInfo.FrameworkType.PYTORCH_LIGHTNING_FRAMEWORK
            framework.version.CopyFrom(self._pl_version)

            if self._check_param(trainer, 'world_size'):
                profile.inference_stats.world_size = trainer.world_size

            profile.model_info.model_format = profiles_pb2.ModelInfo.ModelFormat.PYTORCH_FORMAT
            if self._model_size_mb:
                profile.model_info.model_size_bytes = int(self._model_size_mb * 1e6)

            if self._node_rank >= 0 and graphsignal._agent.node_rank == -1:
                profile.node_usage.node_rank = self._node_rank
            if self._global_rank >= 0 and graphsignal._agent.global_rank == -1:
                profile.process_usage.global_rank = self._global_rank
            if self._local_rank >= 0 and graphsignal._agent.local_rank == -1:
                profile.process_usage.local_rank = self._local_rank
        except Exception as exc:
            self._span._add_profiler_exception(exc)

    def _check_param(self, trainer, param):
        value = getattr(trainer, param, None)
        return isinstance(value, (str, int, float, bool))

    def _log_metrics(self, logged_metrics):
        if logged_metrics:
            for key, value in logged_metrics.items():
                if isinstance(key, str):
                    if isinstance(value, (int, float)):
                        graphsignal.log_metric(key, value)
                    elif isinstance(value, Tensor):
                        graphsignal.log_metric(key, value.item())
