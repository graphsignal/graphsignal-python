"""SDK lifecycle functions used by the watcher process.

`graphsignal-watch` (and any code that configures the watcher process) imports
these. Application code that just wants to start the watcher sidecar should
use `graphsignal.watch()` instead.

Access the active SDK via `graphsignal.sdk.sdk()` (raises if not configured).
"""

from typing import Dict, Optional
import atexit
import logging
import os

from graphsignal.sdk.env_vars import read_config_param, read_config_tags
from graphsignal.sdk.sdk import Sdk

logger = logging.getLogger('graphsignal')

# Module-private singleton. Read via sdk(); not part of the public surface.
_sdk: Optional[Sdk] = None


def configure(
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    deployment: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
    debug_mode: Optional[bool] = None,
    target_pid: Optional[int] = None,
    otel_collector_port: Optional[int] = None,
    metrics_port: Optional[int] = None,
) -> None:
    global _sdk

    if _sdk:
        logger.warning("SDK already configured")
        return

    api_key = read_config_param("api_key", str, api_key, required=True)
    api_base = read_config_param("api_base", str, api_base)
    tags = read_config_tags(tags)
    debug_mode = read_config_param("debug", bool, debug_mode, default_value=False)

    if deployment and isinstance(deployment, str):
        tags['deployment'] = deployment

    if target_pid is None:
        target_pid = os.getpid()

    _sdk = Sdk(
        api_key=api_key,
        api_base=api_base,
        tags=tags,
        debug_mode=debug_mode,
        target_pid=target_pid,
        otel_collector_port=otel_collector_port,
        metrics_port=metrics_port)
    _sdk.setup()

    atexit.register(shutdown)

    logger.debug('SDK configured')


def sdk() -> Sdk:
    """Return the configured SDK singleton, or raise if not configured."""
    if _sdk is None:
        raise RuntimeError('SDK not configured; call graphsignal.sdk.configure() first')
    return _sdk


def is_configured() -> bool:
    """True iff `configure()` has run and `shutdown()` has not."""
    return _sdk is not None


def tick(block: bool = False, force: bool = False) -> None:
    sdk().tick(block=block, force=force)


def shutdown() -> None:
    global _sdk
    if not _sdk:
        return

    atexit.unregister(shutdown)
    _sdk.shutdown()
    _sdk = None

    logger.debug('SDK shutdown')


__all__ = [
    'Sdk',
    'configure',
    'sdk',
    'is_configured',
    'tick',
    'shutdown',
]
