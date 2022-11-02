from typing import Optional
import logging
import transformers
from transformers import TrainerCallback

import graphsignal
from graphsignal.proto import signals_pb2
from graphsignal.proto_utils import parse_semver
from graphsignal.profilers.operation_profiler import OperationProfiler

logger = logging.getLogger('graphsignal')


class GraphsignalCallback(TrainerCallback):
    def __init__(self, tags: Optional[dict] = None, profiler: Optional[OperationProfiler] = None):
        super().__init__()
        self._profiler = profiler
        self._hf_version = None
        self._tags = tags
        self._trace = None
        self._step_idx = -1

    def on_train_begin(self, args, state, control, **kwarg):
        self._configure_profiler(args)

    def on_train_end(self, args, state, control, **kwarg):
        graphsignal.upload()

    def on_step_begin(self, args, state, control, **kwarg):
        self._step_idx += 1
        self._start_trace('training_step', args, state)

    def on_step_end(self, args, state, control, **kwarg):
        self._stop_trace(args, state)

    def _start_trace(self, endpoint, args, state):
        if not self._trace:
            self._trace = graphsignal.start_trace(
                endpoint=endpoint,
                tags=self._tags,
                profiler=self._profiler)
            self._trace.set_tag('step', self._step_idx)

    def _stop_trace(self, args, state):
        if self._trace:
            if self._trace.is_tracing():
                self._update_signal(args, state)
            self._trace.stop()
            self._trace = None

    def _configure_profiler(self, args):
        try:
            self._hf_version = signals_pb2.SemVer()
            parse_semver(self._hf_version, transformers.__version__)
        except Exception:
            logger.error('Error configuring Hugging Face profiler', exc_info=True)

    def _update_signal(self, args, state):
        try:
            signal = self._trace._signal

            signal.agent_info.framework_profiler_type = signals_pb2.AgentInfo.ProfilerType.HUGGING_FACE_PROFILER

            framework = signal.frameworks.add()
            framework.type = signals_pb2.FrameworkInfo.FrameworkType.HUGGING_FACE_FRAMEWORK
            framework.version.CopyFrom(self._hf_version)
        except Exception as exc:
            logger.debug('Error in Hugging Face callback', exc_info=True)
            self._trace._add_profiler_exception(exc)


class GraphsignalPTCallback(GraphsignalCallback):
    def __init__(self, tags: Optional[dict] = None):
        from graphsignal.profilers.pytorch import PyTorchProfiler
        super().__init__(
            tags=tags,
            profiler=PyTorchProfiler())


class GraphsignalTFCallback(GraphsignalCallback):
    def __init__(self, tags: Optional[dict] = None):
        from graphsignal.profilers.tensorflow import TensorflowProfiler
        super().__init__(
            tags=tags,
            profiler=TensorflowProfiler())