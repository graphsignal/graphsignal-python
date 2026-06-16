import logging
import subprocess
import sys
from typing import List, Optional

logger = logging.getLogger('graphsignal')


def extract_port(args: List[str], default: Optional[int] = None) -> Optional[int]:
    """Extract the serving port from engine CLI args (`--port N` / `--port=N`).

    Returns ``default`` when no `--port` is present or the value isn't an int.
    """
    for i, arg in enumerate(args):
        if arg == '--port':
            if i + 1 < len(args):
                try:
                    return int(args[i + 1])
                except (TypeError, ValueError):
                    return default
        elif arg.startswith('--port='):
            try:
                return int(arg.split('=', 1)[1])
            except (TypeError, ValueError):
                return default
    return default


def resolve_metrics_port(metrics_port: Optional[int], args: List[str],
                         default: Optional[int] = None) -> Optional[int]:
    """Pick the Prometheus scrape port for a launcher.

    An explicit `--metrics-port` (forwarded from `graphsignal-run`) always wins;
    otherwise fall back to the engine's serving port (`--port`) or its default.
    """
    if metrics_port is not None:
        return metrics_port
    return extract_port(args, default=default)


def start_watcher(pid: int, otel_collector_port: Optional[int] = None,
                  metrics_port: Optional[int] = None) -> Optional[subprocess.Popen]:
    """Spawn the graphsignal-watch subprocess to observe the given pid.

    Used by `graphsignal.watch()` (pid = self) and by the per-engine launchers,
    which call this just before `execv`-ing the workload (pid = self, which
    becomes the workload's pid after exec).
    """
    cmd = [
        sys.executable,
        '-m', 'graphsignal.commands.graphsignal_watch',
        '--pid', str(int(pid)),
    ]
    if otel_collector_port is not None:
        cmd.extend(['--otel-collector-port', str(int(otel_collector_port))])
    if metrics_port is not None:
        cmd.extend(['--metrics-port', str(int(metrics_port))])

    logger.debug('Starting graphsignal-watch: %s', ' '.join(cmd))
    try:
        return subprocess.Popen(
            cmd,
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            close_fds=True,
        )
    except Exception as exc:
        logger.error('Failed to start graphsignal-watch: %s', exc, exc_info=True)
        return None
