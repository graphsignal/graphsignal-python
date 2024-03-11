from typing import Dict, Any, Union, Optional
import os
import logging
import atexit
import functools
import asyncio
import time

from graphsignal.version import __version__
from graphsignal.tracer import Tracer
from graphsignal.spans import Span, get_current_span, _uuid_sha1, _sanitize_str
from graphsignal.proto import signals_pb2

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

    if not key:
        logger.error('set_tag: key must be provided')
        return

    if value is None:
        _tracer.tags.pop(key, None)
        return

    if len(_tracer.tags) > Span.MAX_RUN_TAGS:
        logger.error('set_tag: too many tags (>{0})'.format(Span.MAX_RUN_TAGS))
        return

    _tracer.tags[key] = value


def get_tag(key: str) -> Optional[str]:
    _check_configured()

    return _tracer.tags.get(key, None)


def set_context_tag(key: str, value: str) -> None:
    _check_configured()

    if not key:
        logger.error('set_context_tag: key must be provided')
        return

    tags = _tracer.context_tags.get()

    if value is None:
        tags.pop(key, None)
        _tracer.context_tags.set(tags)
        return

    if len(tags) > Span.MAX_RUN_TAGS:
        logger.error('set_context_tag: too many tags (>{0})'.format(Span.MAX_RUN_TAGS))
        return

    tags[key] = value
    _tracer.context_tags.set(tags)


def get_context_tag(key: str) -> Optional[str]:
    _check_configured()

    return _tracer.context_tags.get().get(key, None)


def trace(
        operation: str,
        tags: Optional[Dict[str, str]] = None) -> 'Span':
    _check_configured()

    return Span(operation=operation, tags=tags)


def start_trace(
        operation: str,
        tags: Optional[Dict[str, str]] = None) -> 'Span':
    trace(operation, tags)


def trace_function(
        func=None, 
        *,
        operation: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None):
    if func is None:
        return functools.partial(trace_function, operation=operation, tags=tags)

    if operation is None:
        operation_or_name = func.__name__
    else:
        operation_or_name = operation

    if asyncio.iscoroutinefunction(func):
        @functools.wraps(func)
        async def tf_async_wrapper(*args, **kwargs):
            async with trace(operation=operation_or_name, tags=tags):
                return await func(*args, **kwargs)
        return tf_async_wrapper
    else:
        @functools.wraps(func)
        def tf_wrapper(*args, **kwargs):
            with trace(operation=operation_or_name, tags=tags):
                return func(*args, **kwargs)
        return tf_wrapper


def current_span() -> Optional['Span']:
    _check_configured()

    return get_current_span()


def score(
        name: str, 
        tags: Optional[Dict[str, str]] = None,
        score: Optional[Union[int, float]] = None, 
        severity: Optional[int] = None,
        comment: Optional[str] = None) -> None:
    _check_configured()

    now = int(time.time())

    if not name:
        logger.error('score: name is required')
        return

    score_obj = signals_pb2.Score()
    score_obj.score_id = _uuid_sha1(size=12)
    score_obj.name = name

    tag = score_obj.tags.add()
    tag.key = 'deployment'
    tag.value = _tracer.deployment

    if tags:
        for tag_key, tag_value in tags.items():
            tag = score_obj.tags.add()
            tag.key = _sanitize_str(tag_key, max_len=50)
            tag.value = _sanitize_str(tag_value, max_len=250)

    if score is not None:
        score_obj.score = score

    if severity and severity >= 1 and severity <= 5:
        score_obj.severity = severity

    if comment:
        score_obj.comment = comment

    score_obj.create_ts = now

    _tracer.uploader().upload_score(score_obj)
    _tracer.tick(now)


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
