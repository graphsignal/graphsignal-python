from typing import Optional
import logging
import os
import sys
import time
import gzip
import atexit
import threading
import onnxruntime

import graphsignal
from graphsignal.proto_utils import parse_semver
from graphsignal.proto import profiles_pb2
from graphsignal.inference_span import InferenceSpan
from graphsignal.tracers.operation_profiler import OperationProfiler
from graphsignal.tracers.profiler_utils import create_log_dir, remove_log_dir, find_and_read

logger = logging.getLogger('graphsignal')


class ONNXRuntimeProfiler(OperationProfiler):
    def __init__(self):
        self._is_initialized = None
        self._onnx_version = None
        self._first_trace_path = None
        self._end_lock = threading.Lock()

    def reset(self):
        self._first_trace_path = None

    def start(self, profile, onnx_session):
        if not onnx_session.get_session_options().enable_profiling:
            return

        logger.debug('Activating ONNX profiler')

        # Initialization
        if not self._is_initialized:
            self._is_initialized = True

            self._onnx_version = profiles_pb2.SemVer()
            parse_semver(self._onnx_version, onnxruntime.__version__)

        # Profiler info
        profile.profiler_info.operation_profiler_type = profiles_pb2.ProfilerInfo.ProfilerType.ONNX_PROFILER

        # Framework info
        framework = profile.frameworks.add()
        framework.type = profiles_pb2.FrameworkInfo.FrameworkType.ONNX_FRAMEWORK
        framework.version.CopyFrom(self._onnx_version)

    def stop(self, profile, onnx_session):
        if not onnx_session.get_session_options().enable_profiling:
            return

        logger.debug('Deactivating ONNX profiler')

        # end profiling after first inference and use first trace data 
        # in all other profiles for the current inference session
        with self._end_lock:
            if not self._first_trace_path:
                self._first_trace_path = onnx_session.end_profiling()

        if self._first_trace_path:
            trace_file_size = os.path.getsize(self._first_trace_path)
            if trace_file_size > 50 * 1e6:
                raise Exception('Trace file too big: {0}'.format(trace_file_size))

            with open(self._first_trace_path) as f:
                trace_json = f.read()
                profile.trace_data = gzip.compress(trace_json.encode())


class ONNXRuntimeProfilingSession():
    def __init__(self):
        self.log_dir = '{0}{1}'.format(create_log_dir(), os.path.sep)

    def cleanup(self):
        remove_log_dir(self.log_dir)


_profiler = ONNXRuntimeProfiler()
_profiling_sessions = []


def _cleanup_profiling_sessions():
    if _profiling_sessions:
        for profiling_session in _profiling_sessions:
            profiling_session.cleanup()

atexit.register(_cleanup_profiling_sessions)


def initialize_profiler(onnx_session_options: onnxruntime.SessionOptions):
    graphsignal._check_configured()

    profiling_session = ONNXRuntimeProfilingSession()
    _profiling_sessions.append(profiling_session)

    onnx_session_options.enable_profiling = True
    onnx_session_options.profile_file_prefix = profiling_session.log_dir


def inference_span(
        model_name: str,
        metadata: Optional[dict] = None,
        ensure_profile: Optional[bool] = False,
        onnx_session: Optional[onnxruntime.InferenceSession] = None) -> InferenceSpan:
    graphsignal._check_configured()

    if not onnx_session:
        raise ValueError('onnx_session is required')

    return InferenceSpan(
        model_name=model_name,
        metadata=metadata,
        ensure_profile=ensure_profile,
        operation_profiler=_profiler,
        context=onnx_session)
