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

    def _start_profiler(self, span_name, span_type, args, state):
        if not self._span:
            self._span = ProfilingSpan(
                framework_profiler=self._profiler,
                span_name=span_name,
                span_type=span_type)
            if args:
                if args.run_name:
                    self._span.add_metadata('Run name', args.run_name)
            if state:
                if state.epoch is not None:
                    self._span.add_metadata('Epoch', state.epoch)
                if state.global_step is not None:
                    self._span.add_metadata('Global step', state.global_step)

    def _stop_profiler(self):
        if self._span:
            self._span.stop()
            self._span = None

    def on_step_begin(self, args, state, control, **kwarg):
        self._start_profiler(
            'Training step', profiles_pb2.Span.SpanType.TRAINING_STEP, args, state)

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

    def _start_profiler(self, span_name, span_type, args, state):
        if not self._span:
            self._span = ProfilingSpan(
                framework_profiler=self._profiler,
                span_name=span_name,
                span_type=span_type)
            if args:
                if args.run_name:
                    self._span.add_metadata('Run name', args.run_name)
            if state:
                if state.epoch is not None:
                    self._span.add_metadata('Epoch', state.epoch)
                if state.global_step is not None:
                    self._span.add_metadata('Global step', state.global_step)


    def _stop_profiler(self):
        if self._span:
            self._span.stop()
            self._span = None

    def on_step_begin(self, args, state, control, **kwarg):
        self._start_profiler(
            'Training step', profiles_pb2.Span.SpanType.TRAINING_STEP, args, state)

    def on_step_end(self, args, state, control, **kwarg):
        self._stop_profiler()