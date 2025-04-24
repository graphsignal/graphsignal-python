from typing import Dict, Any, Union, Optional, Type
import os
import logging
import atexit

from graphsignal.version import __version__
from graphsignal.tracer import Tracer
from graphsignal.spans import Span, SpanContext

logger = logging.getLogger('graphsignal')

_tracer = None


def _check_configured():
    global _tracer
    if not _tracer:
        raise ValueError(
            'Tracer not configured, call graphsignal.configure() first')


def _parse_env_param(name: str, value: Any, expected_type: Type) -> Any:
    if value is None:
        return None

    try:
        if expected_type == bool:
            return value if isinstance(value, bool) else str(value).lower() in ("true", "1", "yes")
        elif expected_type == int:
            return int(value)
        elif expected_type == float:
            return float(value)
        elif expected_type == str:
            return str(value)
    except (ValueError, TypeError):
        pass

    raise ValueError(f"Invalid type for {name}: expected {expected_type.__name__}, got {type(value).__name__}")


def _read_config_param(name: str, expected_type: Type, provided_value: Optional[Any] = None, default_value: Optional[Any] = None, required: bool = False) -> Any:
    # Check if the value was provided as an argument
    if provided_value is not None:
        return provided_value

    # Check if the value was provided as an environment variable
    env_value = os.getenv(f'GRAPHSIGNAL_{name.upper()}')
    if env_value is not None:
        parsed_env_value = _parse_env_param(name, env_value, expected_type)
        if parsed_env_value is not None:
            return parsed_env_value

    if required:
        raise ValueError(f"Missing required argument: {name}")

    return default_value


def _read_config_tags(provided_value: Optional[dict] = None, prefix: str = "GRAPHSIGNAL_TAG_") -> Dict[str, str]:
    # Check if the value was provided as an argument
    if provided_value is not None:
        return provided_value

    # Check if the value was provided as an environment variable
    return {key[len(prefix):].lower(): value for key, value in os.environ.items() if key.startswith(prefix)}


def configure(
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
    deployment: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
    auto_instrument: Optional[bool] = None,
    profiling_rate: Optional[float] = None,
    debug_mode: Optional[bool] = None
) -> None:
    global _tracer

    if _tracer:
        logger.warning("Tracer already configured")
        return

    api_key = _read_config_param("api_key", str, api_key, required=True)
    api_url = _read_config_param("api_url", str, api_url)
    tags = _read_config_tags(tags)
    auto_instrument = _read_config_param("auto_instrument", bool, auto_instrument, default_value=True)
    profiling_rate = _read_config_param("profiling_rate", float, profiling_rate, default_value=0.1)
    debug_mode = _read_config_param("debug_mode", bool, debug_mode, default_value=False)

    # left for compatibility
    if deployment and isinstance(deployment, str):
        tags['deployment'] = deployment

    _tracer = Tracer(
        api_key=api_key,
        api_url=api_url,
        tags=tags,
        auto_instrument=auto_instrument,
        profiling_rate=profiling_rate,
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
        operation: str,
        tags: Optional[Dict[str, str]] = None,
        with_profile: Optional[bool] = False) -> 'Span':
    _check_configured()

    return _tracer.trace(operation=operation, tags=tags, with_profile=with_profile)


def trace_function(
        func=None, 
        *,
        operation: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        with_profile: Optional[bool] = False) -> Any:
    return _tracer.trace_function(func, operation=operation, tags=tags, with_profile=with_profile)


def score(
        name: str, 
        tags: Optional[Dict[str, str]] = None,
        score: Optional[Union[int, float]] = None, 
        unit: Optional[str] = None,
        severity: Optional[int] = None,
        comment: Optional[str] = None) -> None:
    _check_configured()
    return _tracer.score(
        name=name, 
        tags=tags, 
        score=score, 
        unit=unit, 
        severity=severity, 
        comment=comment)


def upload(block=False) -> None:
    _check_configured()

    _tracer.upload(block=block)


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
    'upload',
    'shutdown',
    'trace',
    'function_trace',
    'SpanContext',
    'Span',
    'score',
    'set_tag',
    'get_tag',
    'set_context_tag',
    'get_context_tag',
    'remove_context_tag'
]
