import logging
import sys

from graphsignal.launchers import auto_flags
from graphsignal.launchers.base_launcher import BaseLauncher
from graphsignal.launchers.command_utils import (
    hash_workload_id, resolve_executable as _resolve, resolve_metrics_port)
from graphsignal.launchers.supervisor import launch_supervised
from graphsignal.profilers.cupti_profiler import CuptiProfiler
from graphsignal.profilers.rocm_profiler import RocmProfiler

logger = logging.getLogger('graphsignal')


class FallbackLauncher(BaseLauncher):
    def match(self) -> bool:
        return True

    def launch(self) -> None:
        if not self.args:
            print('graphsignal-run: no command specified')
            sys.exit(1)

        workload_args = list(self.args)
        workload_id = hash_workload_id(workload_args)

        CuptiProfiler.setup_env_vars(cuda_graph_trace=self.cuda_graph_trace)
        RocmProfiler.setup_env_vars()

        if self.auto_flags:
            self.args = auto_flags.inject_auto_flags(
                self.args, workload_id=workload_id)

        # Generic workloads have no known metrics port; only scrape when the
        # user passed --metrics-port (or the workload itself takes a --port).
        metrics_port = resolve_metrics_port(
            self.metrics_port, self.args, default=None)

        command = self.args[0]
        rest = list(self.args[1:])

        executable = _resolve(command)
        if executable:
            logger.debug('FallbackLauncher launch: %s %s', executable, rest)
            launch_supervised([executable] + rest,
                              workload_id=workload_id,
                              metrics_port=metrics_port)
            return

        # Fall back to `python -m <command>` for module-name targets
        # (e.g. `graphsignal-run mypkg.cli`).
        logger.debug('FallbackLauncher launch python -m: %s', command)
        launch_supervised([sys.executable, '-m', command] + rest,
                          workload_id=workload_id,
                          metrics_port=metrics_port)
