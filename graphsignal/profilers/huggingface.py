import logging
from transformers import TrainerCallback
try:
    from graphsignal.profilers.tensorflow import profile_span as profile_span_tf
except ImportError:
    pass
try:
    from graphsignal.profilers.pytorch import profile_span as profile_span_pt
except ImportError:
    pass

logger = logging.getLogger('graphsignal')


class GraphsignalTFCallback(TrainerCallback):
    __slots__ = [
        '_span'
    ]

    def __init__(self):
        self._span = None

    def _start_profiler(self, span_name, args, state):
        if not self._span:
            self._span = profile_span_pt(span_name=span_name)
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
        self._start_profiler('Training step', args, state)

    def on_step_end(self, args, state, control, **kwarg):
        self._stop_profiler()


class GraphsignalPTCallback(TrainerCallback):
    __slots__ = [
        '_span'
    ]

    def __init__(self):
        self._span = None

    def _start_profiler(self, span_name, args, state):
        if not self._span:
            self._span = profile_span_pt(span_name=span_name)
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
        self._start_profiler('Training step', args, state)

    def on_step_end(self, args, state, control, **kwarg):
        self._stop_profiler()
