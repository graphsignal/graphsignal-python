import logging
import os
import shutil

from graphsignal.launchers.base_launcher import BaseLauncher
from graphsignal.launchers.command_utils import start_watcher
from graphsignal.profilers.cupti_profiler import CuptiProfiler

logger = logging.getLogger('graphsignal')

_TRTLLM_NAMES = {'trtllm', 'trtllm-serve', 'trtllm-llmapi-launch'}


class TrtllmLauncher(BaseLauncher):
    def match(self) -> bool:
        return self.executable_name() in _TRTLLM_NAMES

    def launch(self) -> None:
        # TRT-LLM: CUPTI + watcher only — no argv mutation. The TensorRT
        # backend's `/metrics` is on by default and gets picked up
        # automatically by `PrometheusRecorder` via port scanning.
        CuptiProfiler.setup_env_vars()

        start_watcher(os.getpid(), otel_collector_port=None)

        executable = _resolve(self.args[0])
        if not executable:
            raise FileNotFoundError(f'executable not found: {self.args[0]}')

        logger.debug('TrtllmLauncher exec: %s %s', executable, self.args)
        os.execv(executable, self.args)


def _resolve(name):
    if os.path.isabs(name) and os.path.isfile(name):
        return name
    return shutil.which(name)
