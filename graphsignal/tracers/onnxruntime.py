from typing import Optional
import logging
import os
import gzip
from pathlib import Path
import onnxruntime

import graphsignal
from graphsignal.proto_utils import parse_semver
from graphsignal.proto import signals_pb2
from graphsignal.inference_span import InferenceSpan
from graphsignal.tracers.operation_profiler import OperationProfiler
from graphsignal.tracers.profiler_utils import create_log_dir, remove_log_dir, find_and_read

logger = logging.getLogger('graphsignal')


class ONNXRuntimeProfiler(OperationProfiler):
    def __init__(self):
        self._is_initialized = None
        self._onnx_version = None
        self._ended = False

    def read_info(self, signal):
        if not self._onnx_version:
            self._onnx_version = signals_pb2.SemVer()
            parse_semver(self._onnx_version, onnxruntime.__version__)

        # Framework info
        framework = signal.frameworks.add()
        framework.type = signals_pb2.FrameworkInfo.FrameworkType.ONNX_FRAMEWORK
        framework.version.CopyFrom(self._onnx_version)

    def start(self, signal, onnx_session):
        if not onnx_session.get_session_options().enable_profiling:
            return

        if self._ended:
            return

        logger.debug('Activating ONNX profiler')

        # Profiler info
        signal.agent_info.operation_profiler_type = signals_pb2.AgentInfo.ProfilerType.ONNX_PROFILER

    def stop(self, signal, onnx_session):
        if not onnx_session.get_session_options().enable_profiling:
            return

        if self._ended:
            return
        self._ended = True

        logger.debug('Deactivating ONNX profiler')

        try:
            trace_file_path = onnx_session.end_profiling()
            trace_file_size = os.path.getsize(trace_file_path)
            if trace_file_size > 50 * 1e6:
                raise Exception('Trace file too big: {0}'.format(trace_file_size))

            with open(trace_file_path) as f:
                trace_json = f.read()
                signal.trace_data = gzip.compress(trace_json.encode())
        finally:
            # delete profiler after profiling first inference
            log_dir = str(Path(trace_file_path).parent.absolute())
            remove_log_dir(log_dir)

            key = _extract_session_key(log_dir)
            del _profilers[key]


_profilers = {}


def _extract_session_key(log_dir):
    return os.path.basename(os.path.normpath(log_dir))


def initialize_profiler(onnx_session_options: onnxruntime.SessionOptions):
    graphsignal._check_configured()

    onnx_session_options.enable_profiling = True

    log_dir = '{0}{1}'.format(create_log_dir(), os.path.sep)
    onnx_session_options.profile_file_prefix = log_dir

    key = _extract_session_key(log_dir)
    _profilers[key] = ONNXRuntimeProfiler()


def inference_span(
        model_name: str,
        tags: Optional[dict] = None,
        ensure_trace: Optional[bool] = False,
        onnx_session: Optional[onnxruntime.InferenceSession] = None) -> InferenceSpan:
    graphsignal._check_configured()

    if not onnx_session:
        raise ValueError('onnx_session is required')

    profiler = None
    log_dir = onnx_session.get_session_options().profile_file_prefix
    key = _extract_session_key(log_dir)
    if key in _profilers:
        profiler = _profilers[key]

    return InferenceSpan(
        model_name=model_name,
        tags=tags,
        ensure_trace=ensure_trace,
        operation_profiler=profiler,
        context=onnx_session)
