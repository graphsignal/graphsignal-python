import logging
import os
import shutil
import sys

from graphsignal.launchers import auto_flags
from graphsignal.launchers.base_launcher import BaseLauncher
from graphsignal.launchers.command_utils import resolve_metrics_port, start_watcher
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

        CuptiProfiler.setup_env_vars(cuda_graph_trace=self.cuda_graph_trace)
        RocmProfiler.setup_env_vars()

        if self.auto_flags:
            self.args = auto_flags.inject_auto_flags(self.args)

        # Generic workloads have no known metrics port; only scrape when the
        # user passed --metrics-port (or the workload itself takes a --port).
        metrics_port = resolve_metrics_port(
            self.metrics_port, self.args, default=None)

        start_watcher(os.getpid(), metrics_port=metrics_port)

        command = self.args[0]
        rest = list(self.args[1:])

        executable = _resolve(command)
        if executable:
            logger.debug('FallbackLauncher exec: %s %s', executable, rest)
            try:
                os.execv(executable, [executable] + rest)
            except PermissionError:
                print("graphsignal-run: permission error while launching '%s'" % executable)
                print("Did you mean `graphsignal-run python %s`?" % executable)
                sys.exit(1)
            except Exception as e:
                print("graphsignal-run: error launching '%s': %s" % (executable, e))
                logger.error('error launching executable', exc_info=True)
                raise
            return

        # Fall back to `python -m <command>` for module-name targets
        # (e.g. `graphsignal-run mypkg.cli`).
        logger.debug('FallbackLauncher exec python -m: %s', command)
        os.execv(sys.executable, [sys.executable, '-m', command] + rest)


def _resolve(name):
    if os.path.isabs(name) and os.path.isfile(name):
        return name
    return shutil.which(name)
