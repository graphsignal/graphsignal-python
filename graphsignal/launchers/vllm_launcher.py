import logging
import os
import shutil

from graphsignal.launchers import auto_flags
from graphsignal.launchers.base_launcher import BaseLauncher
from graphsignal.launchers.command_utils import (
    engine_version, resolve_metrics_port, start_watcher)
from graphsignal.otel.otel_collector import OTELCollector
from graphsignal.profilers.cupti_profiler import CuptiProfiler

logger = logging.getLogger('graphsignal')

# vLLM serves Prometheus /metrics on its HTTP server (--port, default 8000).
DEFAULT_SERVE_PORT = 8000


class VllmLauncher(BaseLauncher):
    def match(self) -> bool:
        return self.executable_name() == 'vllm'

    def launch(self) -> None:
        # OTEL trace injection is opt-in via `graphsignal-run --enable-otel`
        # (requires OpenTelemetry installed in the vLLM environment). CUPTI /
        # NVML / process / Prometheus signals flow regardless.
        if not self.enable_otel:
            otel_port = None
            logger.debug('vLLM: OTEL tracing not enabled '
                         '(pass --enable-otel to graphsignal-run to enable)')
        elif _has_flag(self.args, '--otlp-traces-endpoint'):
            # Respect a user-supplied endpoint: no local collector.
            otel_port = None
        else:
            otel_port = OTELCollector.find_port()

        CuptiProfiler.setup_env_vars(cuda_graph_trace=self.cuda_graph_trace)

        if self.auto_flags:
            self.args = auto_flags.inject_auto_flags(
                self.args, engine_name='vllm',
                engine_version=engine_version('vllm'))

        new_args = _inject_vllm_args(self.args, otel_port)

        metrics_port = resolve_metrics_port(
            self.metrics_port, self.args, default=DEFAULT_SERVE_PORT)

        start_watcher(os.getpid(), otel_collector_port=otel_port,
                      metrics_port=metrics_port)

        executable = _resolve(new_args[0])
        if not executable:
            raise FileNotFoundError(f'executable not found: {new_args[0]}')

        logger.debug('VllmLauncher exec: %s %s', executable, new_args)
        os.execv(executable, new_args)


def _inject_vllm_args(args, otel_port):
    args = list(args)

    # Inject our endpoint only when the caller allocated a port (i.e. the
    # user did not supply their own `--otlp-traces-endpoint`).
    if otel_port is not None and not _has_flag(args, '--otlp-traces-endpoint'):
        # Explicit IPv4 loopback (not "localhost") so the exporter and the
        # collector can't disagree on IPv4 vs IPv6.
        args.extend(['--otlp-traces-endpoint', f'127.0.0.1:{otel_port}'])

    # vLLM exposes Prometheus on its HTTP server by default; ensure log stats stay on.
    if _has_flag(args, '--disable-log-stats'):
        args = [a for a in args if a != '--disable-log-stats']

    return args


def _has_flag(args, flag) -> bool:
    for a in args:
        if a == flag or a.startswith(flag + '='):
            return True
    return False


def _resolve(name):
    if os.path.isabs(name) and os.path.isfile(name):
        return name
    return shutil.which(name)
