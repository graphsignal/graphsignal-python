import logging
import os
import sys

from graphsignal.launchers.vllm_launcher import VllmLauncher
from graphsignal.launchers.sglang_launcher import SglangLauncher
from graphsignal.launchers.trtllm_launcher import TrtllmLauncher
from graphsignal.launchers.fallback_launcher import FallbackLauncher

log = logging.getLogger(__name__)


def _setup_logging():
    """Emit the `graphsignal` logger to stderr so launcher diagnostics (which
    launcher matched, the final injected args, the watcher command + OTEL port)
    are visible in the workload's log instead of being silently dropped. The
    launcher runs in the workload process before `execv`, so without this its
    debug output goes nowhere. Verbosity follows GRAPHSIGNAL_DEBUG."""
    debug = os.getenv('GRAPHSIGNAL_DEBUG', '').strip().lower() in ('1', 'true', 'yes')
    gs_logger = logging.getLogger('graphsignal')
    gs_logger.setLevel(logging.DEBUG if debug else logging.WARNING)
    if not any(isinstance(h, logging.StreamHandler) for h in gs_logger.handlers):
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)s graphsignal-run: %(message)s'))
        gs_logger.addHandler(handler)

USAGE = """
Run a target application with the Graphsignal profiler.

Options (must precede the command):
  --enable-otel   Enable OpenTelemetry trace capture for supported engines
                  (injects the engine's trace flags and starts a local OTLP
                  collector). Requires OpenTelemetry installed in the engine's
                  environment. Off by default.
  --metrics-port PORT
                  Port to scrape the engine's Prometheus /metrics endpoint on.
                  Overrides the port derived from the engine's --port/default.
  --cuda-graph-trace graph|node
                  CUDA graph tracing granularity. graph (default): one
                  aggregated cuda.graph event per graph launch, lower overhead.
                  node: per-node kernel/memcpy activity inside graphs, higher
                  overhead.

Example:
  graphsignal-run vllm serve facebook/opt-125m --port 8001
  graphsignal-run --enable-otel sglang serve --model-path <model>
  graphsignal-run --metrics-port 8000 trtllm-serve <model> --port 8000
  graphsignal-run python myapp.py
  graphsignal-run app.py
"""


def _parse_cuda_graph_trace(value):
    mode = (value or '').strip().lower()
    if mode in ('graph', 'node'):
        return mode
    print("graphsignal-run: invalid --cuda-graph-trace value: %s (expected graph or node)\n" % value)
    sys.exit(1)


def _extract_graphsignal_flags(argv):
    """Pull graphsignal-run's own flags out of argv before the workload command.

    `--enable-otel` opts into OTEL trace injection (engine trace flags + local
    collector). 
    `--metrics-port` specifies the Prometheus scrape port.
    `--cuda-graph-trace` sets CUDA graph tracing granularity (graph or node).
    All are consumed here and never forwarded to the workload. Only leading flags
    (before the workload command) are parsed, so identically named workload
    flags later in argv are left untouched.
    """
    enable_otel = False
    metrics_port = None
    cuda_graph_trace = None
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == '--enable-otel':
            enable_otel = True
            i += 1
            continue
        if arg == '--metrics-port':
            if i + 1 >= len(argv):
                print("graphsignal-run: --metrics-port requires a value\n")
                sys.exit(1)
            metrics_port = _parse_port(argv[i + 1])
            i += 2
            continue
        if arg.startswith('--metrics-port='):
            metrics_port = _parse_port(arg.split('=', 1)[1])
            i += 1
            continue
        if arg == '--cuda-graph-trace':
            if i + 1 >= len(argv):
                print("graphsignal-run: --cuda-graph-trace requires a value\n")
                sys.exit(1)
            cuda_graph_trace = _parse_cuda_graph_trace(argv[i + 1])
            i += 2
            continue
        if arg.startswith('--cuda-graph-trace='):
            cuda_graph_trace = _parse_cuda_graph_trace(arg.split('=', 1)[1])
            i += 1
            continue
        break
    return enable_otel, metrics_port, cuda_graph_trace, argv[i:]


def _parse_port(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        print("graphsignal-run: invalid --metrics-port value: %s\n" % value)
        sys.exit(1)


def main():
    if len(sys.argv) < 2:
        print("graphsignal-run: no command specified\n")
        print(USAGE)
        sys.exit(1)

    _setup_logging()

    enable_otel, metrics_port, cuda_graph_trace, target_args = _extract_graphsignal_flags(sys.argv[1:])
    log.debug('graphsignal-run target args: %s (enable_otel=%s, metrics_port=%s, cuda_graph_trace=%s)',
              target_args, enable_otel, metrics_port, cuda_graph_trace)

    launchers = [
        VllmLauncher(target_args, enable_otel=enable_otel, metrics_port=metrics_port,
                     cuda_graph_trace=cuda_graph_trace),
        SglangLauncher(target_args, enable_otel=enable_otel, metrics_port=metrics_port,
                       cuda_graph_trace=cuda_graph_trace),
        TrtllmLauncher(target_args, enable_otel=enable_otel, metrics_port=metrics_port,
                       cuda_graph_trace=cuda_graph_trace),
        FallbackLauncher(target_args, enable_otel=enable_otel, metrics_port=metrics_port,
                         cuda_graph_trace=cuda_graph_trace),
    ]

    for launcher in launchers:
        try:
            matched = launcher.match()
        except Exception as exc:
            log.error('Error matching launcher %s: %s', type(launcher).__name__, exc, exc_info=True)
            continue
        if matched:
            log.debug('Selected launcher: %s', type(launcher).__name__)
            launcher.launch()
            return

    print("graphsignal-run: no launcher matched")
    sys.exit(1)


if __name__ == '__main__':
    main()
