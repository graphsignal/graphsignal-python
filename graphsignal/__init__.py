from typing import Any, Union, Optional
import time
import sys
import os
import logging
import threading
import uuid
import hashlib
import atexit

from graphsignal import version
from graphsignal.agent import Agent
from graphsignal.uploader import Uploader
from graphsignal.usage.process_reader import ProcessReader
from graphsignal.usage.nvml_reader import NvmlReader
from graphsignal.proto import profiles_pb2

logger = logging.getLogger('graphsignal')

_agent = None


def _check_configured():
    global _agent
    if not _agent:
        raise ValueError(
            'Graphsignal profiler not configured, call graphsignal.configure() first')


def _check_and_set_arg(
        name, value, is_str=False, is_int=False, is_bool=False, required=True):
    env_name = 'GRAPHSIGNAL_{0}'.format(name.upper())

    if not value and env_name in os.environ:
        value = os.environ[env_name]
        if value:
            if is_int:
                try:
                    value = int(value)
                except:
                    raise ValueError('Invalid format, expected integer: {0}'.format(name))
            elif is_bool:
                value = bool(value)

    if not value:
        if required:
            raise ValueError('Missing argument: {0}'.format(name))
    else:
        if is_str:
            if not isinstance(value, str):
                raise ValueError('Invalid format, expected string: {0}'.format(name))
        elif is_int:
            if not isinstance(value, int):
                raise ValueError('Invalid format, expected integer: {0}'.format(name))
        elif is_bool:
            if not isinstance(value, bool):
                raise ValueError('Invalid format, expected boolean: {0}'.format(name))

    return value


def configure(
        api_key: str = None,
        workload_name: str = None,
        run_id: Optional[str] = None,
        global_rank: Optional[int] = None,
        node_rank: Optional[int] = None,
        local_rank: Optional[int] = None,
        debug_mode: Optional[bool] = False,
        disable_fwk_profiler: Optional[bool] = False) -> None:
    global _agent

    if _agent:
        logger.warning('Graphsignal profiler already configured')
        return

    debug_mode = _check_and_set_arg('debug_mode', debug_mode, is_bool=True, required=False)
    if debug_mode:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.WARNING)
    api_key = _check_and_set_arg('api_key', api_key, is_str=True, required=True)
    workload_name = _check_and_set_arg('workload_name', workload_name,  is_str=True, required=True)
    run_id = _check_and_set_arg('run_id', run_id, is_str=True, required=False)
    global_rank = _check_and_set_arg('global_rank', global_rank, is_int=True, required=False)
    node_rank = _check_and_set_arg('node_rank', node_rank, is_int=True, required=False)
    local_rank = _check_and_set_arg('local_rank', local_rank, is_int=True, required=False)
    disable_fwk_profiler = _check_and_set_arg('disable_fwk_profiler', disable_fwk_profiler, is_bool=True, required=False)

    if not run_id:
        run_id = _uuid_sha1()
        # set run ID to be picked up by worker processes
        os.environ['GRAPHSIGNAL_RUN_ID'] = run_id
        logger.debug('run_id (or GRAPHSIGNAL_RUN_ID) not available, generated: %s', run_id)
    else:
        logger.debug('run_id (or GRAPHSIGNAL_RUN_ID) available: %s', run_id)

    _agent = Agent()
    _agent.worker_id = _uuid_sha1(size=12)
    _agent.start_ms = int(time.time() * 1e3)
    _agent.api_key = api_key
    _agent.workload_name = workload_name[:50]
    _agent.run_id = _sha1(run_id, size=12)
    _agent.node_rank = node_rank if node_rank is not None else -1
    _agent.local_rank = local_rank if local_rank is not None else -1
    _agent.global_rank = global_rank if global_rank is not None else -1
    _agent.debug_mode = debug_mode
    _agent.uploader = Uploader()
    _agent.uploader.configure()
    _agent.process_reader = ProcessReader()
    _agent.process_reader.setup()
    _agent.nvml_reader = NvmlReader()
    _agent.nvml_reader.setup()

    atexit.register(shutdown)

    logger.debug('Graphsignal profiler configured')


def add_tag(tag: str) -> None:
    _check_configured()

    if tag is None or not isinstance(tag, str):
        raise ValueError('Missing or invalid argument: tag')

    global _agent
    if _agent.tags is None:
        _agent.tags = {}
    _agent.tags[tag[:50]] = True


def log_parameter(name: str, value: Any) -> None:
    _check_configured()

    if name is None or not isinstance(name, str):
        raise ValueError('Missing or invalid argument: name')

    if value is None:
        raise ValueError('Missing argument: value')

    global _agent
    if _agent.params is None:
        _agent.params = {}
    _agent.params[name[:250]] = str(value)[:1000]


def log_metric(name: str, value: Union[int, float]) -> None:
    _check_configured()

    if name is None or not isinstance(name, str):
        raise ValueError('Missing or invalid argument: name')

    if value is None or not isinstance(value, (int, float)):
        raise ValueError('Missing argument: value')

    if _agent.metrics is None:
        _agent.metrics = {}
    _agent.metrics[name[:250]] = value


def shutdown() -> None:
    _check_configured()

    global _agent
    atexit.unregister(shutdown)
    _agent.uploader.flush()
    _agent.process_reader.shutdown()
    _agent.nvml_reader.shutdown()
    _agent = None

    logger.debug('Graphsignal profiler shutdown')


def generate_uuid() -> None:
    return _uuid_sha1()    


def _sha1(text, size=-1):
    sha1_hash = hashlib.sha1()
    sha1_hash.update(text.encode('utf-8'))
    return sha1_hash.hexdigest()[0:size]


def _uuid_sha1(size=-1):
    return _sha1(str(uuid.uuid4()), size)
