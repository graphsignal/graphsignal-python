import logging
import os
import shutil

from graphsignal.launchers import auto_flags
from graphsignal.launchers.base_launcher import BaseLauncher
from graphsignal.launchers.command_utils import (
    engine_version, resolve_metrics_host, resolve_metrics_port, start_watcher)
from graphsignal.profilers.cupti_profiler import CuptiProfiler
from graphsignal.profilers.rocm_profiler import RocmProfiler

logger = logging.getLogger('graphsignal')

_TRTLLM_NAMES = {'trtllm', 'trtllm-serve', 'trtllm-llmapi-launch'}

# trtllm-serve exposes Prometheus metrics on its HTTP server (--port, default 8000).
DEFAULT_SERVE_PORT = 8000
DEFAULT_SERVE_HOST = 'localhost'
# Separate from `/metrics`, which returns JSON iteration stats (IterationStats).
DEFAULT_METRICS_PATH = '/prometheus/metrics'


class TrtllmLauncher(BaseLauncher):
    def match(self) -> bool:
        return self.executable_name() in _TRTLLM_NAMES

    def launch(self) -> None:
        if _has_flag(self.args, '--grpc'):
            logger.warning(
                'TRT-LLM gRPC mode has no HTTP /prometheus/metrics endpoint; '
                'engine Prometheus metrics will not be scraped')

        CuptiProfiler.setup_env_vars(cuda_graph_trace=self.cuda_graph_trace)
        RocmProfiler.setup_env_vars()

        if self.auto_flags:
            self.args = auto_flags.inject_auto_flags(
                self.args, engine_name='tensorrt-llm',
                engine_version=engine_version('tensorrt_llm'))

        metrics_port = resolve_metrics_port(
            self.metrics_port, self.args, default=DEFAULT_SERVE_PORT)
        metrics_host = resolve_metrics_host(
            None, self.args, default=DEFAULT_SERVE_HOST)

        start_watcher(os.getpid(), otel_collector_port=None,
                      metrics_port=metrics_port,
                      metrics_path=DEFAULT_METRICS_PATH,
                      metrics_host=metrics_host)

        executable = _resolve(self.args[0])
        if not executable:
            raise FileNotFoundError(f'executable not found: {self.args[0]}')

        logger.debug('TrtllmLauncher exec: %s %s', executable, self.args)
        os.execv(executable, self.args)


def _has_flag(args, flag) -> bool:
    for a in args:
        if a == flag or a.startswith(flag + '='):
            return True
    return False


def _resolve(name):
    if os.path.isabs(name) and os.path.isfile(name):
        return name
    return shutil.which(name)
