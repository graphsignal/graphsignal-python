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
from graphsignal.profilers.operation_profiler import OperationProfiler
from graphsignal.profilers.profiler_utils import create_log_dir, remove_log_dir, find_and_read

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
            try:
                if os.path.getsize(self._first_trace_path) > 50 * 1e6:
                    raise Exception('Trace file too big')

                with open(self._first_trace_path) as f:
                    trace_json = f.read()
                    profile.trace_data = gzip.compress(trace_json.encode())
            except Exception as e:
                logger.error('Error exporting Chrome trace', exc_info=True)


class ONNXRuntimeProfilingSession():
    def __init__(self):
        self.log_dir = '{0}{1}'.format(create_log_dir(), os.path.sep)

    def cleanup(self):
        remove_log_dir(self.log_dir)


_profiler = ONNXRuntimeProfiler()
_profiling_session = None


def _cleanup_profiling_session():
    global _profiling_session

    if _profiling_session:
        _profiling_session.cleanup()

atexit.register(_cleanup_profiling_session)


def initialize_profiler(session_options: onnxruntime.SessionOptions):
    graphsignal._check_configured()

    global _profiling_session

    if _profiling_session:
        _profiler.reset()
        _profiling_session.cleanup()

    _profiling_session = ONNXRuntimeProfilingSession()

    session_options.enable_profiling = True
    session_options.profile_file_prefix = _profiling_session.log_dir


def profile_inference(
        session: onnxruntime.InferenceSession,
        batch_size: Optional[int] = None,
        ensure_profile: Optional[bool] = False) -> InferenceSpan:
    graphsignal._check_configured()

    return InferenceSpan(
        batch_size=batch_size,
        ensure_profile=ensure_profile,
        operation_profiler=_profiler,
        context=session)
