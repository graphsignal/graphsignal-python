from typing import Dict, Any, Union, Optional, Type
import logging
import atexit

from graphsignal.version import __version__
from graphsignal.env_vars import read_config_param, read_config_tags
from graphsignal.core.ticker import Ticker
from graphsignal.signals.spans import Span, SpanContext
from graphsignal.bootstrap.utils import add_bootstrap_to_pythonpath

logger = logging.getLogger('graphsignal')

_ticker = None


def _check_configured():
    global _ticker
    if not _ticker:
        raise ValueError(
            'SDK not configured, call graphsignal.configure() first')

def configure(
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
    deployment: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
    auto_instrument: Optional[bool] = None,
    debug_mode: Optional[bool] = None
) -> None:
    global _ticker

    if _ticker:
        logger.warning("SDK already configured")
        return

    api_key = read_config_param("api_key", str, api_key, required=True)
    api_url = read_config_param("api_url", str, api_url)
    tags = read_config_tags(tags)
    auto_instrument = read_config_param("auto_instrument", bool, auto_instrument, default_value=True)
    debug_mode = read_config_param("debug_mode", bool, debug_mode, default_value=False)

    # left for compatibility
    if deployment and isinstance(deployment, str):
        tags['deployment'] = deployment

    _ticker = Ticker(
        api_key=api_key,
        api_url=api_url,
        tags=tags,
        auto_instrument=auto_instrument,
        debug_mode=debug_mode)
    _ticker.setup()

    add_bootstrap_to_pythonpath()

    atexit.register(shutdown)

    logger.debug('SDK configured')


def set_tag(key: str, value: str) -> None:
    _check_configured()
    _ticker.set_tag(key, value)


def get_tag(key: str) -> Optional[str]:
    _check_configured()
    return _ticker.get_tag(key)


def remove_tag(key: str):
    _check_configured()
    return _ticker.remove_tag(key)


def set_context_tag(key: str, value: str, append_uuid=None) -> None:
    _check_configured()
    _ticker.set_context_tag(key, value, append_uuid=append_uuid)


def get_context_tag(key: str) -> Optional[str]:
    _check_configured()
    return _ticker.get_context_tag(key)


def remove_context_tag(key: str):
    _check_configured()
    return _ticker.remove_context_tag(key)


def trace(
        span_name: str,
        tags: Optional[Dict[str, str]] = None) -> 'Span':
    _check_configured()

    return _ticker.trace(span_name=span_name, tags=tags)


def trace_function(
        func=None, 
        *,
        span_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None) -> Any:
    return _ticker.trace_function(func, span_name=span_name, tags=tags)


def profile_function(
        func=None, 
        *,
        category: Optional[str] = None,
        event_name: Optional[str] = None) -> Any:
    _check_configured()
    return _ticker.profile_function(func, category=category, event_name=event_name)


def profile_function_path(
        path: str,
        category: Optional[str] = None,
        event_name: Optional[str] = None) -> Any:
    _check_configured()
    return _ticker.profile_function_path(path, category=category, event_name=event_name)


def profile_cuda_kernel(kernel_pattern: str, event_name: str) -> None:
    _check_configured()
    return _ticker.profile_cuda_kernel(kernel_pattern, event_name)


def log_message(message: str, *, tags: Optional[Dict[str, str]] = None, level: Optional[str] = None, exception: Optional[str] = None) -> None:
    _check_configured()
    return _ticker.log_message(message=message, tags=tags, level=level, exception=exception)


def tick(block=False, force=False) -> None:
    _check_configured()

    _ticker.tick(block=block, force=force)


def shutdown() -> None:
    global _ticker
    if not _ticker:
        return

    atexit.unregister(shutdown)
    _ticker.shutdown()
    _ticker = None

    logger.debug('SDK shutdown')


__all__ = [
    '__version__',
    'configure',
    'tick',
    'shutdown',
    'trace',
    'trace_function',
    'profile_function',
    'profile_function_path',
    'profile_cuda_kernel',
    'SpanContext',
    'Span',
    'log_message',
    'set_tag',
    'get_tag',
    'set_context_tag',
    'get_context_tag',
    'remove_context_tag'
]
