from typing import Dict, Any, Union, Optional, Type
import os
import logging
import atexit

from graphsignal.version import __version__
from graphsignal.env_vars import read_config_param, read_config_tags
from graphsignal.tracer import Tracer
from graphsignal.spans import Span, SpanContext

logger = logging.getLogger('graphsignal')

_tracer = None


def _check_configured():
    global _tracer
    if not _tracer:
        raise ValueError(
            'Tracer not configured, call graphsignal.configure() first')

def configure(
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
    deployment: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
    auto_instrument: Optional[bool] = None,
    samples_per_min: Optional[int] = None,
    include_profiles: Optional[list] = None,
    debug_mode: Optional[bool] = None
) -> None:
    global _tracer

    if _tracer:
        logger.warning("Tracer already configured")
        return

    api_key = read_config_param("api_key", str, api_key, required=True)
    api_url = read_config_param("api_url", str, api_url)
    tags = read_config_tags(tags)
    auto_instrument = read_config_param("auto_instrument", bool, auto_instrument, default_value=True)
    samples_per_min = read_config_param("samples_per_min", int, samples_per_min)
    include_profiles = read_config_param("include_profiles", list, include_profiles)
    debug_mode = read_config_param("debug_mode", bool, debug_mode, default_value=False)

    # left for compatibility
    if deployment and isinstance(deployment, str):
        tags['deployment'] = deployment

    _tracer = Tracer(
        api_key=api_key,
        api_url=api_url,
        tags=tags,
        auto_instrument=auto_instrument,
        samples_per_min=samples_per_min,
        include_profiles=include_profiles,
        debug_mode=debug_mode)
    _tracer.setup()

    atexit.register(shutdown)

    logger.debug('Tracer configured')


def set_tag(key: str, value: str) -> None:
    _check_configured()
    _tracer.set_tag(key, value)


def get_tag(key: str) -> Optional[str]:
    _check_configured()
    return _tracer.get_tag(key)


def remove_tag(key: str):
    _check_configured()
    return _tracer.remove_tag(key)


def set_context_tag(key: str, value: str, append_uuid=None) -> None:
    _check_configured()
    _tracer.set_context_tag(key, value, append_uuid=append_uuid)


def get_context_tag(key: str) -> Optional[str]:
    _check_configured()
    return _tracer.get_context_tag(key)


def remove_context_tag(key: str):
    _check_configured()
    return _tracer.remove_context_tag(key)


def set_param(name: str, value: Any) -> None:
    _check_configured()
    _tracer.set_param(name, value)


def get_param(name: str) -> Optional[Any]:
    _check_configured()
    return _tracer.get_param(name)


def remove_param(name: str):
    _check_configured()
    return _tracer.remove_param(name)


def trace(
        span_name: str,
        tags: Optional[Dict[str, str]] = None,
        include_profiles: Optional[list] = None) -> 'Span':
    _check_configured()

    return _tracer.trace(span_name=span_name, tags=tags, include_profiles=include_profiles)


def trace_function(
        func=None, 
        *,
        span_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        include_profiles: Optional[list] = None) -> Any:
    return _tracer.trace_function(func, span_name=span_name, tags=tags, include_profiles=include_profiles)


def report_error(
        name: str, 
        tags: Optional[Dict[str, str]] = None,
        level: Optional[str] = None,
        message: Optional[str] = None,
        exc_info: Optional[tuple] = None) -> None:
    _check_configured()
    return _tracer.report_error(
        name=name, 
        tags=tags,
        level=level, 
        message=message,
        exc_info=exc_info)


def tick(block=False, force=False) -> None:
    _check_configured()

    _tracer.tick(block=block, force=force)


def shutdown() -> None:
    global _tracer
    if not _tracer:
        return

    atexit.unregister(shutdown)
    _tracer.shutdown()
    _tracer = None

    logger.debug('Tracer shutdown')


__all__ = [
    '__version__',
    'configure',
    'tick',
    'shutdown',
    'trace',
    'trace_function',
    'SpanContext',
    'Span',
    'report_error',
    'set_tag',
    'get_tag',
    'set_context_tag',
    'get_context_tag',
    'remove_context_tag'
]
