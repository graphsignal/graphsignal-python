import logging

from pytorch_lightning.callbacks.base import Callback

import graphsignal
from graphsignal.proto import profiles_pb2
from graphsignal.profilers.pytorch import PyTorchProfiler
from graphsignal.profiling_step import ProfilingStep

logger = logging.getLogger('graphsignal')

TRAINER_FLAGS = [
    'accelerator',
    'devices',
    'accumulate_grad_batches',
    'amp_backend',
    'amp_level',
    'auto_scale_batch_size',
    'auto_scale_batch_size',
    'auto_lr_find',
    'benchmark',
    'deterministic',
    'check_val_every_n_epoch',
    'enable_checkpointing',
    'fast_dev_run',
    'gradient_clip_val',
    'limit_train_batches',
    'limit_test_batches',
    'limit_val_batches',
    'log_every_n_steps',
    'max_epochs',
    'min_epochs',
    'max_steps',
    'min_steps',
    'max_time',
    'num_nodes',
    'num_sanity_val_steps',
    'overfit_batches',
    'precision',
    'enable_progress_bar',
    'reload_dataloaders_every_n_epochs',
    'replace_sampler_ddp',
    'strategy',
    'sync_batchnorm',
    'tpu_cores',
    'val_check_interval'
]

class GraphsignalCallback(Callback):
    __slots__ = [
        '_profiler',
        '_step'
    ]

    def __init__(self, effective_batch_size=None):
        self._profiler = PyTorchProfiler()
        self._step = None

        self._effective_batch_size = effective_batch_size
        super().__init__()

    def _start_profiler(self, run_phase, trainer):
        if not self._step:
            self._step = ProfilingStep(
                run_phase=run_phase,
                effective_batch_size=self._effective_batch_size,
                framework_profiler=self._profiler)

    def _stop_profiler(self, trainer):
        if self._step:
            if self._step._is_scheduled:
                if hasattr(trainer, 'accumulate_grad_batches') and isinstance(trainer.accumulate_grad_batches, int):
                    self._step._profile.step_stats.gradient_accumulation_steps = trainer.accumulate_grad_batches
            self._step.stop()
            self._step = None

    def on_train_start(self, trainer, pl_module):
        _add_parameters_from_trainer(trainer)

    def on_validation_start(self, trainer, pl_module):
        _add_parameters_from_trainer(trainer)

    def on_test_start(self, trainer, pl_module):
        _add_parameters_from_trainer(trainer)

    def on_predict_start(self, trainer, pl_module):
        _add_parameters_from_trainer(trainer)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self._start_profiler(profiles_pb2.RunPhase.TRAINING, trainer)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._stop_profiler(trainer)

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

def _add_parameters_from_trainer(trainer):
    trainer_vars = vars(trainer)
    for flag in TRAINER_FLAGS:
        if flag in trainer_vars:
            value = trainer_vars[flag]
            if isinstance(value, (str, int, float, bool)):
                graphsignal.add_parameter(flag, value)