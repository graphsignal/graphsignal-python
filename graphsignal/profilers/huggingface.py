import logging
from transformers import TrainerCallback

from graphsignal.proto import profiles_pb2
from graphsignal.profiling_span import ProfilingSpan

logger = logging.getLogger('graphsignal')


class GraphsignalPTCallback(TrainerCallback):
    __slots__ = [
        '_profiler',
        '_span'
    ]

    def __init__(self):
        from graphsignal.profilers.pytorch import PyTorchProfiler
        self._profiler = PyTorchProfiler()
        self._span = None

    def _start_profiler(self, run_phase, args, state):
        if not self._span:
            self._span = ProfilingSpan(
                run_phase=run_phase,
                is_step=True,
                framework_profiler=self._profiler)

    def _stop_profiler(self):
        if self._span:
            self._span.stop()
            self._span = None

    def on_step_begin(self, args, state, control, **kwarg):
        self._start_profiler(profiles_pb2.RunPhase.TRAINING, args, state)

    def on_step_end(self, args, state, control, **kwarg):
        self._stop_profiler()


class GraphsignalTFCallback(TrainerCallback):
    __slots__ = [
        '_profiler',
        '_span'
    ]

    def __init__(self):
        from graphsignal.profilers.tensorflow import TensorflowProfiler
        self._profiler = TensorflowProfiler()
        self._span = None

    def _start_profiler(self, run_phase, args, state):
        if not self._span:
            self._span = ProfilingSpan(
                run_phase=run_phase,
                is_step=True,
                framework_profiler=self._profiler)

    def _stop_profiler(self):
        if self._span:
            self._span.stop()
            self._span = None

    def on_step_begin(self, args, state, control, **kwarg):
        self._start_profiler(profiles_pb2.RunPhase.TRAINING, args, state)

    def on_step_end(self, args, state, control, **kwarg):
        self._stop_profiler()