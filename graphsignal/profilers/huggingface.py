import logging
from transformers import TrainerCallback

import graphsignal
from graphsignal.proto import profiles_pb2
from graphsignal.profiling_step import ProfilingStep

logger = logging.getLogger('graphsignal')

PHASE_TRAINING = 'training'

EXCLUDE_ARGS = {
    'logging_dir',
    'local_rank'
}

class GraphsignalCallback(TrainerCallback):
    def __init__(self, framework_profiler):
        super().__init__()
        self._profiler = framework_profiler
        self._step = None
        self._local_rank = -1

    def on_train_begin(self, args, state, control, **kwarg):
        self._configure_profiler(args)

    def on_train_end(self, args, state, control, **kwarg):
        self._stop_profiler(args, state)

    def on_step_begin(self, args, state, control, **kwarg):
        self._stop_profiler(args, state)
        self._start_profiler(PHASE_TRAINING, args, state)

    def on_step_end(self, args, state, control, **kwarg):
        pass

    def _start_profiler(self, phase_name, args, state):
        if not self._step:
            self._step = ProfilingStep(
                phase_name=phase_name,
                effective_batch_size=self._get_effective_batch_size(args),
                framework_profiler=self._profiler)

    def _stop_profiler(self, args, state):
        if self._step:
            if self._step._is_scheduled:
                self._update_profile(args, state)
            self._step.stop()
            self._step = None

    def _get_effective_batch_size(self, args):
        gradient_accumulation_steps = args.gradient_accumulation_steps if args.gradient_accumulation_steps > 0 else 1
        return args.train_batch_size * args.gradient_accumulation_steps

    def _update_profile(self, args, state):
        profile = self._step._profile

        if args.local_rank == -1 or args.local_rank == 0:
            profile.step_stats.flop_count = _uint(state.total_flos)
        profile.step_stats.batch_size = args.train_batch_size
        profile.step_stats.device_batch_size = args.per_device_train_batch_size

        if self._local_rank >= 0 and graphsignal._agent.local_rank == -1:
            profile.process_usage.local_rank = self._local_rank

    def _configure_profiler(self, args):
        if args.local_rank >= 0:
            self._local_rank = args.local_rank

        if args.world_size > 0:
            graphsignal.log_parameter('world_size', args.world_size)

        for name, value in vars(args).items():
            if not name.startswith('_') and name not in EXCLUDE_ARGS and isinstance(value, (str, int, float, bool)):
                graphsignal.log_parameter(name, value)


def _uint(val):
    return max(int(val), 0)


class GraphsignalPTCallback(GraphsignalCallback):
    def __init__(self):
        from graphsignal.profilers.pytorch import PyTorchProfiler
        super().__init__(PyTorchProfiler())


class GraphsignalTFCallback(GraphsignalCallback):
    def __init__(self):
        from graphsignal.profilers.tensorflow import TensorflowProfiler
        super().__init__(TensorflowProfiler())
