import logging
from transformers import TrainerCallback

import graphsignal
from graphsignal.proto import profiles_pb2
from graphsignal.profiling_step import ProfilingStep
from graphsignal import step_counter

logger = logging.getLogger('graphsignal')

EXCLUDE_ARGS = {
    'logging_dir',
    'local_rank'
}

class GraphsignalPTCallback(TrainerCallback):
    __slots__ = [
        '_profiler',
        '_step'
    ]

    def __init__(self):
        from graphsignal.profilers.pytorch import PyTorchProfiler
        self._profiler = PyTorchProfiler()
        self._step = None

    def on_train_begin(elf, args, state, control, **kwarg):
        step_counter.init_step_stats(profiles_pb2.RunPhase.TRAINING)
        _configure_profiler(args)

    def on_train_end(elf, args, state, control, **kwarg):
        step_counter.reset_step_stats(profiles_pb2.RunPhase.TRAINING)

    def on_step_begin(self, args, state, control, **kwarg):
        self._start_profiler(profiles_pb2.RunPhase.TRAINING, args, state)

    def on_step_end(self, args, state, control, **kwarg):
        self._stop_profiler(args, state)

    def _start_profiler(self, run_phase, args, state):
        if not self._step:
            self._step = ProfilingStep(
                run_phase=run_phase,
                effective_batch_size=_get_effective_batch_size(args),
                framework_profiler=self._profiler)

    def _stop_profiler(self, args, state):
        if self._step:
            if self._step._is_scheduled:
                _fill_step_stats(self._step, args, state)
            self._step.stop()
            self._step = None


class GraphsignalTFCallback(TrainerCallback):
    __slots__ = [
        '_profiler',
        '_step'
    ]

    def __init__(self):
        from graphsignal.profilers.tensorflow import TensorflowProfiler
        self._profiler = TensorflowProfiler()
        self._step = None

    def on_train_begin(elf, args, state, control, **kwarg):
        step_counter.init_step_stats(profiles_pb2.RunPhase.TRAINING)
        _configure_profiler(args)

    def on_train_end(elf, args, state, control, **kwarg):
        step_counter.reset_step_stats(profiles_pb2.RunPhase.TRAINING)

    def on_step_begin(self, args, state, control, **kwarg):
        self._start_profiler(profiles_pb2.RunPhase.TRAINING, args, state)

    def on_step_end(self, args, state, control, **kwarg):
        self._stop_profiler(args, state)

    def _start_profiler(self, run_phase, args, state):
        if not self._step:
            self._step = ProfilingStep(
                run_phase=run_phase,
                effective_batch_size=_get_effective_batch_size(args),
                framework_profiler=self._profiler)

    def _stop_profiler(self, args, state):
        if self._step:
            if self._step._is_scheduled:
                _fill_step_stats(self._step, args, state)
            self._step.stop()
            self._step = None


def _get_effective_batch_size(args):
    world_size = args.world_size if args.world_size > 0 else 1
    gradient_accumulation_steps = args.gradient_accumulation_steps if args.gradient_accumulation_steps > 0 else 1
    return args.train_batch_size * args.gradient_accumulation_steps * world_size


def _fill_step_stats(step, args, state):
    step_stats = step._profile.step_stats
    step_stats.total_flops = state.total_flos
    step_stats.batch_size = args.train_batch_size
    step_stats.device_batch_size = args.per_device_train_batch_size


def _configure_profiler(args):
    if args.local_rank >= 0:
        graphsignal._agent.local_rank = args.local_rank

    if args.world_size > 0:
        graphsignal.log_parameter('world_size', args.world_size)

    for name, value in vars(args).items():
        if not name.startswith('_') and name not in EXCLUDE_ARGS and isinstance(value, (str, int, float, bool)):
            graphsignal.log_parameter(name, value)


def _log_args_prop(args, prop):
    value = getattr(args, prop, None)
    if isinstance(value, (str, int, float, bool)):
        graphsignal.log_parameter(prop, value)
