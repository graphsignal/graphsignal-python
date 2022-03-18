import logging
from transformers import TrainerCallback

import graphsignal

logger = logging.getLogger('graphsignal')


class GraphsignalTFCallback(TrainerCallback):
    __slots__ = [
        '_span'
    ]

    def __init__(self):
        self._span = None

    def _start_profiler(self, span_name, batch=None):
        if not self._span:
            self._span = graphsignal.profile_span_tf(span_name=span_name)
            if batch is not None:
                self._span.add_metadata('batch', batch)

    def _stop_profiler(self):
        if self._span:
            self._span.stop()
            self._span = None

    def on_step_begin(self, args, state, control, **kwarg):
        self._start_profiler('Training step')

    def on_step_end(self, args, state, control, **kwarg):
        self._stop_profiler()


class GraphsignalPTCallback(TrainerCallback):
    __slots__ = [
        '_span'
    ]

    def __init__(self):
        self._span = None

    def _start_profiler(self, span_name, batch=None):
        if not self._span:
            self._span = graphsignal.profile_span_pt(span_name=span_name)
            if batch is not None:
                self._span.add_metadata('batch', batch)

    def _stop_profiler(self):
        if self._span:
            self._span.stop()
            self._span = None

    def on_step_begin(self, args, state, control, **kwarg):
        self._start_profiler('Training step')

    def on_step_end(self, args, state, control, **kwarg):
        self._stop_profiler()
