from typing import Optional
import logging
import time
from torch import Tensor
import pytorch_lightning
from pytorch_lightning.callbacks.base import Callback

import graphsignal
from graphsignal.proto import signals_pb2
from graphsignal.proto_utils import parse_semver, add_framework_param

logger = logging.getLogger('graphsignal')


class GraphsignalCallback(Callback):
    def __init__(self, tags: Optional[dict] = None):
        super().__init__()

        self._is_initialized = False
        self._framework = None
        self._model_info = None
        self._rank = None
        self._local_rank = None
        self._node_rank = None
        self._trace = None
        self._tags = tags
        self._model_size_mb = None

    def on_predict_start(self, trainer, pl_module):
        self._configure(trainer, pl_module)

    def on_predict_end(self, trainer, pl_module):
        graphsignal.upload()

    def on_predict_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        self._start_trace('predict_batch', batch_idx)

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self._stop_trace()

    def _configure(self, trainer, pl_module):
        try:
            if not self._is_initialized:
                self._is_initialized = True

                self._framework = signals_pb2.FrameworkInfo()
                self._framework.type = signals_pb2.FrameworkInfo.FrameworkType.PYTORCH_LIGHTNING_FRAMEWORK
                parse_semver(self._framework.version, pytorch_lightning.__version__)

                if self._check_param(trainer, 'world_size') and trainer.world_size >= 0:
                    add_framework_param(self._framework, 'world_size', trainer.world_size)
                if self._check_param(trainer, 'node_rank') and trainer.node_rank >= 0:
                    add_framework_param(self._framework, 'node_rank', trainer.node_rank)
                    self._node_rank = trainer.node_rank
                if self._check_param(trainer, 'local_rank') and trainer.local_rank >= 0:
                    add_framework_param(self._framework, 'local_rank', trainer.local_rank)
                    self._local_rank = trainer.local_rank
                if self._check_param(trainer, 'global_rank') and trainer.global_rank >= 0:
                    add_framework_param(self._framework, 'global_rank', trainer.global_rank)
                    self._rank = trainer.global_rank

                self._model_info = signals_pb2.ModelInfo()
                self._model_info.model_format = signals_pb2.ModelInfo.ModelFormat.PYTORCH_FORMAT
                model_size_mb = pytorch_lightning.utilities.memory.get_model_size_mb(pl_module)
                if model_size_mb:
                    self._model_info.model_size_bytes = int(model_size_mb * 1e6)
        except Exception:
            logger.error('Error configuring PyTorch Lightning callback', exc_info=True)

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
            if self._model_info:
                signal.model_info.CopyFrom(self._model_info)
            if self._rank is not None:
                signal.process_usage.rank = self._rank
                signal.process_usage.has_rank = True
            if self._local_rank is not None:
                signal.process_usage.local_rank = self._local_rank
                signal.process_usage.has_local_rank = True
            if self._node_rank is not None:
                signal.node_usage.node_rank = self._node_rank
                signal.node_usage.has_node_rank = True

        except Exception as exc:
            logger.debug('Error in Hugging Face callback', exc_info=True)
            self._trace._add_agent_exception(exc)

    def _check_param(self, trainer, param):
        value = getattr(trainer, param, None)
        return isinstance(value, (str, int, float, bool))