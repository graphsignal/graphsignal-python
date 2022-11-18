from typing import Optional
import logging
import transformers
from transformers import TrainerCallback

import graphsignal
from graphsignal.proto import signals_pb2
from graphsignal.proto_utils import parse_semver, add_framework_param

logger = logging.getLogger('graphsignal')


class GraphsignalCallback(TrainerCallback):
    def __init__(self, tags: Optional[dict] = None):
        super().__init__()
        self._is_initialized = False
        self._framework = None
        self._tags = tags
        self._trace = None
        self._step_idx = -1

    def on_train_begin(self, args, state, control, **kwarg):
        self._configure(args)

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
                tags=self._tags)
            self._trace.set_tag('step', self._step_idx)

    def _stop_trace(self, args, state):
        if self._trace:
            if self._trace.is_sampling():
                self._update_signal(args, state)
            self._trace.stop()
            self._trace = None

    def _configure(self, args):
        try:
            if not self._is_initialized:
                self._is_initialized = True

                self._framework = signals_pb2.FrameworkInfo()
                self._framework.type = signals_pb2.FrameworkInfo.FrameworkType.HUGGING_FACE_FRAMEWORK
                parse_semver(self._framework.version, transformers.__version__)

                if args.local_rank >= 0:
                    add_framework_param(self._framework, 'world_size', args.world_size)
                    add_framework_param(self._framework, 'local_rank', args.local_rank)
        except Exception:
            logger.error('Error configuring Hugging Face callback', exc_info=True)

    def _update_signal(self, args, state):
        try:
            signal = self._trace._signal

            if self._framework:
                signal.frameworks.append(self._framework)
        except Exception as exc:
            logger.debug('Error in Hugging Face callback', exc_info=True)
            self._trace._add_agent_exception(exc)
