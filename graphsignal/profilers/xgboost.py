from typing import Optional
import logging

import xgboost as xgb
from xgboost.callback import TrainingCallback

import graphsignal
from graphsignal.proto import profiles_pb2
from graphsignal.proto_utils import parse_semver
from graphsignal.profilers.generic import GenericProfiler
from graphsignal.profiling_step import ProfilingStep

logger = logging.getLogger('graphsignal')

PHASE_TRAINING = 'training'


class GraphsignalCallback(TrainingCallback):
    def __init__(self):
        super().__init__()
        self._xgboost_version = None
        self._profiler = GenericProfiler()
        self._step = None

    def before_training(self, model):
        self._configure_profiler()
        return model

    def after_training(self, model):
        self._stop_profiler()
        return model

    def before_iteration(self, model, epoch, evals_log):
        self._stop_profiler()
        self._start_profiler(PHASE_TRAINING)
        return False

    def after_iteration(self, model, epoch, evals_log):
        self._log_metrics(evals_log)
        return False

    def _configure_profiler(self):
        try:
            self._xgboost_version = profiles_pb2.SemVer()
            parse_semver(self._xgboost_version, xgb.__version__)
        except Exception:
            logger.error('Error configuring XGBoost profiler', exc_info=True)

    def _start_profiler(self, phase_name):
        if not self._step:
            self._step = ProfilingStep(
                phase_name=phase_name,
                operation_profiler=self._profiler)

    def _stop_profiler(self):
        if self._step:
            if self._step._is_scheduled:
                self._update_profile()
            self._step.stop()
            self._step = None

    def _update_profile(self):
        try:
            profile = self._step._profile

            profile.profiler_info.framework_profiler_type = profiles_pb2.ProfilerInfo.ProfilerType.XGBOOST_PROFILER

            framework = profile.frameworks.add()
            framework.type = profiles_pb2.FrameworkInfo.FrameworkType.XGBOOST_FRAMEWORK
            framework.version.CopyFrom(self._xgboost_version)
        except Exception as exc:
            self._step._add_profiler_exception(exc)

    def _log_metrics(self, evals_log):
        for data_name, metrics in evals_log.items():
            for metric_name, log in metrics.items():
                if len(log) > 0:
                    value = log[-1]
                    if isinstance(value, (int, float)):
                        graphsignal.log_metric(
                            '{0}-{1}'.format(data_name, metric_name), 
                            value)
