import logging

try:
    from pytorch_lightning.profiler import BaseProfiler as Profiler
except ImportError:
    from pytorch_lightning.profiler import Profiler as Profiler

import graphsignal
from graphsignal.profilers.pytorch import profile_span

logger = logging.getLogger('graphsignal')


class GraphsignalProfiler(Profiler):
    __slots__ = [
        '_span'
    ]

    def __init__(self):
        self._span = None
        super().__init__()

    def _start_profiler(self, span_name):
        if not self._span:
            self._span = profile_span(span_name=span_name)

    def _stop_profiler(self):
        if self._span:
            self._span.stop()
            self._span = None

    def start(self, action_name: str) -> None:
        if action_name == 'training_step':
            self._start_profiler('Training step')
        elif action_name == 'validation_step':
            self._start_profiler('Validation step')
        elif action_name == 'test_step':
            self._start_profiler('Test step')
        elif action_name == 'predict_step':
            self._start_profiler('Prediction step')

    def stop(self, action_name: str) -> None:
        self._stop_profiler()

    def summary(self):
        pass

    def teardown(self, stage):
        pass