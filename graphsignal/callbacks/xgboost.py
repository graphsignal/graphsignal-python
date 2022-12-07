from typing import Optional
import logging
import time
import xgboost as xgb
from xgboost.callback import TrainingCallback

import graphsignal
from graphsignal.proto import signals_pb2
from graphsignal.proto_utils import parse_semver, add_framework_param

logger = logging.getLogger('graphsignal')


class GraphsignalCallback(TrainingCallback):
    def __init__(self, tags: Optional[dict] = None):
        super().__init__()

        self._is_initialized = False
        self._framework = None
        self._model_info = None
        self._trace = None
        self._tags = tags
        self._model_size_mb = None

    def before_training(self, model):
        self._configure(model)
        return model

    def after_training(self, model):
        return model

    def before_iteration(self, model, epoch, evals_log):
        self._start_trace('iteration', epoch)
        return False

    def after_iteration(self, model, epoch, evals_log):
        self._stop_trace()
        return False

    def _configure(self, model):
        try:
            if not self._is_initialized:
                self._is_initialized = True
        except Exception:
            logger.error('Error configuring XGBoost callback', exc_info=True)

    def _start_trace(self, endpoint, epoch):
        if not self._trace:
            self._trace = graphsignal.start_trace(
                endpoint=endpoint,
                tags=self._tags)
            self._trace.set_tag('epoch', epoch)

    def _stop_trace(self):
        if self._trace:
            if self._trace.is_sampling():
                self._update_signal()
            self._trace.stop()
            self._trace = None

    def _update_signal(self):
        try:
            signal = self._trace._signal

            if self._framework:
                signal.frameworks.append(self._framework)

            if self._model_info:
                signal.model_info.CopyFrom(self._model_info)

        except Exception as exc:
            logger.debug('Error in Hugging Face callback', exc_info=True)
            self._trace._add_agent_exception(exc)

    def _check_param(self, trainer, param):
        value = getattr(trainer, param, None)
        return isinstance(value, (str, int, float, bool))
