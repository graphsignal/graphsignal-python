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
from graphsignal.span_scheduler import SpanScheduler
from graphsignal.profiling_span import ProfilingSpan
from graphsignal.usage.host_reader import HostReader
from graphsignal.usage.nvml_reader import NvmlReader

logger = logging.getLogger('graphsignal')

_agent = None


def _check_configured():
    global _agent
    if not _agent:
        raise ValueError(
            'Graphsignal profiler not configured, call graphsignal.configure() first')


def configure(api_key, workload_name, debug_mode=False):
    global _agent

    if _agent:
        logger.warning('Graphsignal profiler already configured')
        return

    if debug_mode:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.WARNING)

    if not api_key:
        raise ValueError('Missing argument: api_key')

    if not workload_name:
        raise ValueError('Missing argument: workload_name')

    _agent = Agent()
    _agent.api_key = api_key
    _agent.run_id = _uuid_sha1(size=12)
    _agent.run_start_ms = int(time.time() * 1e3)
    _agent.workload_name = workload_name
    _agent.debug_mode = debug_mode
    _agent.uploader = Uploader()
    _agent.uploader.configure()
    _agent.host_reader = HostReader()
    _agent.host_reader.setup()
    _agent.nvml_reader = NvmlReader()
    _agent.nvml_reader.setup()

    atexit.register(shutdown)

    logger.debug('Graphsignal profiler configured')


def shutdown():
    _check_configured()

    global _agent
    atexit.unregister(shutdown)
    _agent.uploader.flush()
    _agent.host_reader.shutdown()
    _agent.nvml_reader.shutdown()
    _agent = None

    logger.debug('Graphsignal profiler shutdown')


def profile_span_tf(span_name=None, ensure_profile=False):
    _check_configured()

    if not _agent.span_scheduler:
        _agent.span_scheduler = SpanScheduler()

    if not _agent.profiler:
        from graphsignal.profilers.tensorflow_profiler import TensorflowProfiler
        _agent.profiler = TensorflowProfiler()

    return ProfilingSpan(
        scheduler=_agent.span_scheduler,
        profiler=_agent.profiler,
        span_name=span_name,
        ensure_profile=ensure_profile)


def profile_span_pt(span_name=None, ensure_profile=False):
    _check_configured()

    if not _agent.span_scheduler:
        _agent.span_scheduler = SpanScheduler()

    if not _agent.profiler:
        from graphsignal.profilers.pytorch_profiler import PytorchProfiler
        _agent.profiler = PytorchProfiler()

    return ProfilingSpan(
        scheduler=_agent.span_scheduler,
        profiler=_agent.profiler,
        span_name=span_name,
        ensure_profile=ensure_profile)


def _sha1(text, size=-1):
    sha1_hash = hashlib.sha1()
    sha1_hash.update(text.encode('utf-8'))
    return sha1_hash.hexdigest()[0:size]


def _uuid_sha1(size=-1):
    return _sha1(str(uuid.uuid4()), size)
