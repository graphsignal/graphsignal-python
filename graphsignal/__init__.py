from typing import Any, Union, Optional
import os
import logging
import atexit

from graphsignal.version import __version__
from graphsignal.agent import Agent

logger = logging.getLogger('graphsignal')

_agent = None


def _check_configured():
    global _agent
    if not _agent:
        raise ValueError(
            'Agent not configured, call graphsignal.configure() first')


def _check_and_set_arg(
        name, value, is_str=False, is_int=False, is_bool=False, required=False, max_len=None):
    env_name = 'GRAPHSIGNAL_{0}'.format(name.upper())

    if not value and env_name in os.environ:
        value = os.environ[env_name]
        if value:
            if is_int:
                try:
                    value = int(value)
                except:
                    raise ValueError('configure: invalid format, expected integer: {0}'.format(name))
            elif is_bool:
                value = bool(value)

    if not value:
        if required:
            raise ValueError('configure: missing argument: {0}'.format(name))
    else:
        if is_str:
            if not isinstance(value, str):
                raise ValueError('configure: invalid format, expected string: {0}'.format(name))
            if max_len and len(value) > max_len:
                raise ValueError('configure: invalid format, string too long (>{0}): {1}'.format(name, max_len))
        elif is_int:
            if not isinstance(value, int):
                raise ValueError('configure: invalid format, expected integer: {0}'.format(name))
        elif is_bool:
            if not isinstance(value, bool):
                raise ValueError('configure: invalid format, expected boolean: {0}'.format(name))

    return value


def configure(
        api_key: Optional[str] = None,
        debug_mode: Optional[bool] = False) -> None:
    global _agent

    if _agent:
        logger.warning('Agent already configured')
        return

    debug_mode = _check_and_set_arg('debug_mode', debug_mode, is_bool=True)
    if debug_mode:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.WARNING)
    api_key = _check_and_set_arg('api_key', api_key, is_str=True, required=True)

    _agent = Agent(
        api_key=api_key, 
        debug_mode=debug_mode)
    _agent.start()

    atexit.register(shutdown)

    logger.debug('Agent configured')


def tracer(with_profiler: Union[bool, str] = True):
    _check_configured()

    return _agent.tracer(with_profiler=with_profiler)


def upload(block=False) -> None:
    _check_configured()

    _agent.upload(block=block)


def shutdown() -> None:
    global _agent
    _check_configured()

    atexit.unregister(shutdown)
    _agent.stop()
    _agent = None

    logger.debug('Agent shutdown')


__all__ = [
    '__version__',
    'configure',
    'tracer'
    'upload',
    'shutdown'
]