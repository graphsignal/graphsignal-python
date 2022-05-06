import logging

from pytorch_lightning.callbacks.base import Callback

import graphsignal
from graphsignal.proto import profiles_pb2
from graphsignal.profilers.pytorch import PyTorchProfiler
from graphsignal.profiling_step import ProfilingStep
from graphsignal import step_counter

logger = logging.getLogger('graphsignal')


class GraphsignalCallback(Callback):
    __slots__ = [
        '_profiler',
        '_step',
        '_batch_size'
    ]

    def __init__(self, batch_size=None):
        self._profiler = PyTorchProfiler()
        self._step = None
        self._batch_size = batch_size
        super().__init__()

    def on_train_start(self, trainer, pl_module):
        step_counter.init_step_stats(profiles_pb2.RunPhase.TRAINING)
        self._configure_profiler(trainer)

    def on_train_end(self, trainer, pl_module):
        step_counter.reset_step_stats(profiles_pb2.RunPhase.TRAINING)

    def on_validation_start(self, trainer, pl_module):
        step_counter.init_step_stats(profiles_pb2.RunPhase.VALIDATION)
        self._configure_profiler(trainer)

    def on_validation_end(self, trainer, pl_module):
        step_counter.reset_step_stats(profiles_pb2.RunPhase.VALIDATION)

    def on_test_start(self, trainer, pl_module):
        step_counter.init_step_stats(profiles_pb2.RunPhase.TEST)
        self._configure_profiler(trainer)

    def on_test_end(self, trainer, pl_module):
        step_counter.reset_step_stats(profiles_pb2.RunPhase.TEST)

    def on_predict_start(self, trainer, pl_module):
        step_counter.init_step_stats(profiles_pb2.RunPhase.PREDICTION)
        self._configure_profiler(trainer)

    def on_predict_end(self, trainer, pl_module):
        step_counter.reset_step_stats(profiles_pb2.RunPhase.PREDICTION)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self._start_profiler(profiles_pb2.RunPhase.TRAINING, trainer)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._stop_profiler(trainer)
        step_stats = step_counter.get_step_stats(profiles_pb2.RunPhase.TRAINING)

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        self._start_profiler(profiles_pb2.RunPhase.VALIDATION, trainer)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self._stop_profiler(trainer)

    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        self._start_profiler(profiles_pb2.RunPhase.TEST, trainer)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self._stop_profiler(trainer)

    def on_predict_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        self._start_profiler(profiles_pb2.RunPhase.PREDICTION, trainer)

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self._stop_profiler(trainer)

    def _configure_profiler(self, trainer):
        if self._check_param(trainer, 'node_rank') and trainer.node_rank >= 0:
            graphsignal._agent.node_rank = trainer.node_rank
        if self._check_param(trainer, 'local_rank') and trainer.local_rank >= 0:
            graphsignal._agent.local_rank = trainer.local_rank
        if self._check_param(trainer, 'global_rank') and trainer.global_rank >= 0:
            graphsignal._agent.world_rank = trainer.global_rank

        if self._batch_size:
            graphsignal.log_parameter('batch_size', self._batch_size)

        self._log_basic_param(trainer, 'auto_scale_batch_size')
        self._log_basic_param(trainer, 'accumulate_grad_batches')
        self._log_basic_param(trainer, 'world_size')
        self._log_basic_param(trainer, 'num_nodes')
        self._log_basic_param(trainer, 'num_devices')
        self._log_basic_param(trainer, 'precision')
        self._log_basic_param(trainer, 'data_parallel')
        self._log_basic_param(trainer, 'max_epochs')
        self._log_basic_param(trainer, 'min_epochs')
        self._log_basic_param(trainer, 'max_steps')
        self._log_basic_param(trainer, 'min_steps')

    def _start_profiler(self, run_phase, trainer):
        if not self._step:
            self._step = ProfilingStep(
                run_phase=run_phase,
                effective_batch_size=self._batch_size,
                framework_profiler=self._profiler)

    def _stop_profiler(self, trainer):
        if self._step:
            if self._step._is_scheduled:
                self._update_profile(trainer)
            self._step.stop()
            self._step = None

    def _update_profile(self, trainer):
        step_stats = self._step._profile.step_stats

        if self._batch_size:
            step_stats.batch_size = self._batch_size
            strategy = getattr(trainer, 'strategy', None)
            if strategy:
                try:
                    from pytorch_lightning.strategies import DataParallelStrategy, DDP2Strategy
                    if isinstance(strategy, (DataParallelStrategy, DDP2Strategy)):
                        num_devices = getattr(trainer, 'num_devices', 1)
                        if isinstance(num_devices, int) and num_devices > 1:
                            step_stats.device_batch_size = int(self._batch_size / num_devices)
                except:
                    logger.warning('Error recording per-device batch size', exc_info=True)

        if self._check_param(trainer, 'world_size'):
            step_stats.world_size = trainer.world_size

    def _log_basic_param(self, trainer, param):
        value = getattr(trainer, param, None)
        if isinstance(value, (str, int, float, bool)):
            graphsignal.log_parameter(param, value)

    def _check_param(self, trainer, param):
        value = getattr(trainer, param, None)
        return isinstance(value, (str, int, float, bool))
