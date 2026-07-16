import logging
import os
import shutil

from graphsignal.launchers import auto_flags
from graphsignal.launchers.base_launcher import BaseLauncher
from graphsignal.launchers.command_utils import (
    engine_version, hash_workload_id, resolve_metrics_port, start_watcher)
from graphsignal.otel.otel_collector import OTELCollector
from graphsignal.profilers.cupti_profiler import CuptiProfiler
from graphsignal.profilers.rocm_profiler import RocmProfiler

logger = logging.getLogger('graphsignal')

_SGLANG_NAMES = {'sglang', 'sglang.launch_server'}

# SGLang serves Prometheus /metrics on its HTTP server (--port, default 30000).
DEFAULT_SERVE_PORT = 30000


class SglangLauncher(BaseLauncher):
    def match(self) -> bool:
        name = self.executable_name()
        if name in _SGLANG_NAMES:
            return True
        # `python -m sglang.launch_server …`
        if len(self.args) >= 3 and os.path.basename(self.args[0]).startswith('python') \
                and self.args[1] == '-m' and self.args[2] in _SGLANG_NAMES:
            return True
        return False

    def launch(self) -> None:
        workload_args = list(self.args)
        workload_id = hash_workload_id(workload_args)

        # OTEL trace injection is opt-in via `graphsignal-run --enable-otel`.
        # It requires OpenTelemetry installed in the SGLang environment
        # (SGLang >=0.5.10 raises at startup with --enable-trace if it's
        # missing). CUPTI / NVML / process / Prometheus signals flow regardless.
        if not self.enable_otel:
            otel_port = None
            logger.debug('SGLang: OTEL tracing not enabled '
                         '(pass --enable-otel to graphsignal-run to enable)')
        elif _has_flag(self.args, '--otlp-traces-endpoint'):
            # Respect a user-supplied endpoint: don't allocate a collector port
            # and the watcher won't start an OTEL receiver.
            otel_port = None
            logger.info('SGLang: user-supplied --otlp-traces-endpoint detected; '
                        'not starting a local OTEL collector')
        else:
            otel_port = OTELCollector.find_port()
            logger.info('SGLang: allocated local OTEL collector port %s; '
                        'injecting --enable-trace + --otlp-traces-endpoint 127.0.0.1:%s',
                        otel_port, otel_port)

        CuptiProfiler.setup_env_vars(cuda_graph_trace=self.cuda_graph_trace)
        RocmProfiler.setup_env_vars()

        if self.auto_flags:
            self.args = auto_flags.inject_auto_flags(
                self.args, engine_name='sglang',
                engine_version=engine_version('sglang'),
                workload_id=workload_id)

        new_args = _inject_sglang_args(self.args, otel_port, self.enable_otel)

        metrics_port = resolve_metrics_port(
            self.metrics_port, self.args, default=DEFAULT_SERVE_PORT)

        start_watcher(os.getpid(), workload_id=workload_id,
                      otel_collector_port=otel_port,
                      metrics_port=metrics_port)

        executable = _resolve(new_args[0])
        if not executable:
            raise FileNotFoundError(f'executable not found: {new_args[0]}')

        logger.debug('SglangLauncher exec: %s %s', executable, new_args)
        os.execv(executable, new_args)


def _inject_sglang_args(args, otel_port, enable_otel):
    args = list(args)

    # Metrics are independent of OpenTelemetry (Prometheus), so always enable.
    if not _has_flag(args, '--enable-metrics'):
        args.append('--enable-metrics')

    # Tracing is opt-in (graphsignal-run --enable-otel).
    if enable_otel and not _has_flag(args, '--enable-trace'):
        args.append('--enable-trace')

    if otel_port is not None and not _has_flag(args, '--otlp-traces-endpoint'):
        # Explicit IPv4 loopback (not "localhost") so the exporter and the
        # collector can't disagree on IPv4 vs IPv6.
        args.extend(['--otlp-traces-endpoint', f'127.0.0.1:{otel_port}'])

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
