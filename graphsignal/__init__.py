from typing import Dict, Any, Union, Optional
import os
import logging
import atexit
import functools
import asyncio
import time

from graphsignal.version import __version__
from graphsignal.tracer import Tracer
from graphsignal.spans import Span
from graphsignal import client

logger = logging.getLogger('graphsignal')

_tracer = None


def _check_configured():
    global _tracer
    if not _tracer:
        raise ValueError(
            'Tracer not configured, call graphsignal.configure() first')


def _check_and_set_arg(
        name, value, is_str=False, is_int=False, is_bool=False, is_kv=False, required=False, max_len=None):
    env_name = 'GRAPHSIGNAL_{0}'.format(name.upper())

    if not value and env_name in os.environ:
        value = os.environ[env_name]
        if value:
            if is_str:
                if max_len and len(value) > max_len:
                    raise ValueError('configure: invalid format, expected string with max length {0}: {1}'.format(max_len, name))
            if is_int:
                try:
                    value = int(value)
                except:
                    raise ValueError('configure: invalid format, expected integer: {0}'.format(name))
            elif is_bool:
                value = bool(value)
            elif is_kv:
                try:
                    value = dict([el.strip(' ') for el in kv.split('=')] for kv in value.split(','))
                except:
                    raise ValueError('configure: invalid format, expected comma-separated key-value list (k1=v1,k2=v2): {0}'.format(name))

    if not value and required:
        raise ValueError('configure: missing argument: {0}'.format(name))

    return value


def configure(
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        deployment: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        auto_instrument: Optional[bool] = True,
        record_payloads: Optional[bool] = True,
        upload_on_shutdown: Optional[bool] = True,
        debug_mode: Optional[bool] = False) -> None:
    global _tracer

    if _tracer:
        logger.warning('Tracer already configured')
        return

    debug_mode = _check_and_set_arg('debug_mode', debug_mode, is_bool=True)
    api_key = _check_and_set_arg('api_key', api_key, is_str=True, required=True)
    api_url = _check_and_set_arg('api_url', api_url, is_str=True, required=False)
    deployment = _check_and_set_arg('deployment', deployment, is_str=True, required=True)
    tags = _check_and_set_arg('tags', tags, is_kv=True, required=False)

    _tracer = Tracer(
        api_key=api_key,
        api_url=api_url,
        deployment=deployment,
        tags=tags,
        auto_instrument=auto_instrument,
        record_payloads=record_payloads,
        upload_on_shutdown=upload_on_shutdown,
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


def set_context_tag(key: str, value: str) -> None:
    _check_configured()
    _tracer.set_context_tag(key, value)


def get_context_tag(key: str) -> Optional[str]:
    _check_configured()
    return _tracer.get_context_tag(key)


def trace(
        operation: str,
        tags: Optional[Dict[str, str]] = None) -> 'Span':
    _check_configured()

    return _tracer.trace(operation=operation, tags=tags)


def start_trace(
        operation: str,
        tags: Optional[Dict[str, str]] = None) -> 'Span':
    trace(operation, tags)


def trace_function(
        func=None, 
        *,
        operation: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None):
    return _tracer.trace_function(func, operation=operation, tags=tags)


def current_span() -> Optional['Span']:
    _check_configured()

    return _tracer.current_span()


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
    'start_trace',
    'trace',
    'function_trace',
    'Span',
    'score',
    'callbacks'
]
