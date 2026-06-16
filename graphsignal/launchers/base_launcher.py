import logging
import os
from abc import ABC, abstractmethod
from typing import List, Optional

logger = logging.getLogger('graphsignal')


class BaseLauncher(ABC):
    def __init__(self, args: List[str], enable_otel: bool = False,
                 metrics_port: Optional[int] = None):
        self.args: List[str] = list(args)
        # OTEL trace injection (engine --enable-trace / --otlp-traces-endpoint
        # + local collector) is opt-in via `graphsignal-run --enable-otel`.
        self.enable_otel: bool = enable_otel
        # Explicit Prometheus scrape port from `graphsignal-run --metrics-port`.
        # When None, each launcher derives it from the engine's --port/default.
        self.metrics_port: Optional[int] = metrics_port

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
