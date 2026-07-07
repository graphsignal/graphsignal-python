"""Public package entry point.

The only user-facing function here is `graphsignal.watch()`, which spawns the
`graphsignal-watch` sidecar to observe the calling process. Everything that
runs inside the watcher lives in `graphsignal.sdk` and is accessed via
`graphsignal.sdk.sdk()`.
"""

import logging
import os
import subprocess
from typing import Optional

from graphsignal.version import __version__
from graphsignal.launchers.command_utils import start_watcher as _start_watcher
from graphsignal.profilers.cupti_profiler import CuptiProfiler as _CuptiProfiler
from graphsignal.profilers.rocm_profiler import RocmProfiler as _RocmProfiler

logger = logging.getLogger('graphsignal')


def watch(otel_collector_port: Optional[int] = None,
          metrics_port: Optional[int] = None,
          metrics_path: Optional[str] = None,
          metrics_host: Optional[str] = None,
          cuda_graph_trace: Optional[str] = None) -> Optional[subprocess.Popen]:
    """Spawn the `graphsignal-watch` sidecar to observe the current process.

    1. Sets up the CUPTI env vars (`CUDA_INJECTION64_PATH`, `LD_LIBRARY_PATH`)
       so the injection library can attach when CUDA initializes in this
       process.
    2. Spawns `graphsignal-watch --pid <self_pid> [--otel-collector-port ...]
       [--metrics-port ...] [--metrics-path ...]`. The watcher reads its
       `api_key` / `api_base` / tags / etc. from the `GRAPHSIGNAL_*`
       environment variables. Pass `metrics_port` to scrape a Prometheus
       metrics endpoint on that port (`metrics_path` defaults to `/metrics`).
       Pass `cuda_graph_trace` (`'graph'` or `'node'`) to control CUDA graph
       tracing granularity (default: graph-level, lower overhead).

    Returns the watcher `Popen` so the caller can `wait()` or `terminate()`.
    """
    _CuptiProfiler.setup_env_vars(cuda_graph_trace=cuda_graph_trace)
    _RocmProfiler.setup_env_vars()
    return _start_watcher(os.getpid(), otel_collector_port=otel_collector_port,
                          metrics_port=metrics_port, metrics_path=metrics_path,
                          metrics_host=metrics_host)


__all__ = [
    '__version__',
    'watch',
]
