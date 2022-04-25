import logging
from transformers import TrainerCallback

import graphsignal
from graphsignal.proto import profiles_pb2
from graphsignal.profiling_step import ProfilingStep

logger = logging.getLogger('graphsignal')


class GraphsignalPTCallback(TrainerCallback):
    __slots__ = [
        '_profiler',
        '_step'
    ]

    def __init__(self):
        from graphsignal.profilers.pytorch import PyTorchProfiler
        self._profiler = PyTorchProfiler()
        self._step = None

    def _start_profiler(self, run_phase, args, state):
        if not self._step:
            self._step = ProfilingStep(
                run_phase=run_phase,
                framework_profiler=self._profiler)

    def _stop_profiler(self, args, state):
        if self._step:
            if self._step._is_scheduled:
                _fill_step_stats(self._step._profile.step_stats, args, state)
            self._step.stop()
            self._step = None

    def on_train_begin(elf, args, state, control, **kwarg):
        _add_parameters_from_args(args)

    def on_step_begin(self, args, state, control, **kwarg):
        self._start_profiler(profiles_pb2.RunPhase.TRAINING, args, state)

    def on_step_end(self, args, state, control, **kwarg):
        self._stop_profiler(args, state)


class GraphsignalTFCallback(TrainerCallback):
    __slots__ = [
        '_profiler',
        '_step'
    ]

    def __init__(self):
        from graphsignal.profilers.tensorflow import TensorflowProfiler
        self._profiler = TensorflowProfiler()
        self._step = None

    def _start_profiler(self, run_phase, args, state):
        if not self._step:
            self._step = ProfilingStep(
                run_phase=run_phase,
                framework_profiler=self._profiler)

    def _stop_profiler(self, args, state):
        if self._step:
            if self._step._is_scheduled:
                _fill_step_stats(self._step._profile.step_stats, args, state)
            self._step.stop()
            self._step = None

    def on_train_begin(elf, args, state, control, **kwarg):
        _add_parameters_from_args(args)

    def on_step_begin(self, args, state, control, **kwarg):
        self._start_profiler(profiles_pb2.RunPhase.TRAINING, args, state)

    def on_step_end(self, args, state, control, **kwarg):
        self._stop_profiler(args, state)


def _fill_step_stats(step_stats, args, state):
    step_stats.total_flops = state.total_flos
    if args.gradient_accumulation_steps > 0:
        step_stats.batch_size = args.train_batch_size * args.gradient_accumulation_steps
        step_stats.device_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps
    else:
        step_stats.batch_size = args.train_batch_size
        step_stats.device_batch_size = args.per_device_train_batch_size
    step_stats.gradient_accumulation_steps = args.gradient_accumulation_steps

exclude_args = {
    'logging_dir'
}
def _add_parameters_from_args(args):
    for name, value in vars(args).items():
        if name not in exclude_args and isinstance(value, (str, int, float, bool)):
            graphsignal.add_parameter(name, value)