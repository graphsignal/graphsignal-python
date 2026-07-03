import logging
import os
from abc import ABC, abstractmethod
from typing import List, Optional

logger = logging.getLogger('graphsignal')


class BaseLauncher(ABC):
    def __init__(self, args: List[str], enable_otel: bool = False,
                 metrics_port: Optional[int] = None,
                 cuda_graph_trace: Optional[str] = None,
                 auto_flags: bool = False):
        self.args: List[str] = list(args)
        # OTEL trace injection (engine --enable-trace / --otlp-traces-endpoint
        # + local collector) is opt-in via `graphsignal-run --enable-otel`.
        self.enable_otel: bool = enable_otel
        # Explicit Prometheus scrape port from `graphsignal-run --metrics-port`.
        # When None, each launcher derives it from the engine's --port/default.
        self.metrics_port: Optional[int] = metrics_port
        # CUDA graph tracing granularity from `graphsignal-run --cuda-graph-trace`
        # or `graphsignal.watch(cuda_graph_trace=...)`. When None, the native
        # injection lib defaults to graph-level tracing.
        self.cuda_graph_trace: Optional[str] = cuda_graph_trace
        # Fetch recommended engine flags from Graphsignal and merge them into
        # the engine command line, opt-in via `graphsignal-run --auto-flags`.
        self.auto_flags: bool = auto_flags

    @abstractmethod
    def match(self) -> bool:
        ...

    @abstractmethod
    def launch(self) -> None:
        ...

    def executable_name(self) -> str:
        if not self.args:
            return ''
        return os.path.basename(self.args[0])
