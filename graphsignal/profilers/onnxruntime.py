from typing import Optional
import logging
import os
import gzip
from pathlib import Path
import onnxruntime

import graphsignal
from graphsignal.proto_utils import parse_semver
from graphsignal.proto import signals_pb2
from graphsignal.profilers.operation_profiler import OperationProfiler
from graphsignal.profilers.profiler_utils import create_log_dir, remove_log_dir, find_and_read

logger = logging.getLogger('graphsignal')


class ONNXRuntimeProfiler(OperationProfiler):
    def __init__(self):
        self._onnx_version = None
        self._onnx_session = None
        self._log_dir = None
        self._ended = False

    def read_info(self, signal):
        if not self._onnx_version:
            self._onnx_version = signals_pb2.SemVer()
            parse_semver(self._onnx_version, onnxruntime.__version__)

        # Framework info
        framework = signal.frameworks.add()
        framework.type = signals_pb2.FrameworkInfo.FrameworkType.ONNX_FRAMEWORK
        framework.version.CopyFrom(self._onnx_version)

    def initialize_options(self, onnx_session_options: onnxruntime.SessionOptions):
        graphsignal._check_configured()

        onnx_session_options.enable_profiling = True

        self._log_dir = '{0}{1}'.format(create_log_dir(), os.path.sep)
        onnx_session_options.profile_file_prefix = self._log_dir

    def set_onnx_session(self, onnx_session):
        self._onnx_session = onnx_session

    def start(self, signal):
        if not self._log_dir:
            raise ValueError('ONNX Runtime profiler is not initialized')

        if not self._onnx_session:
            raise ValueError('ONNX Runtime session is not provided')

        if not self._onnx_session.get_session_options().enable_profiling:
            raise ValueError('ONNX Runtime profiling is not enabled')

        if self._ended:
            return

        logger.debug('Activating ONNX profiler')

        # Profiler info
        signal.agent_info.operation_profiler_type = signals_pb2.AgentInfo.ProfilerType.ONNX_PROFILER

    def stop(self, signal):
        if self._ended:
            return
        self._ended = True

        logger.debug('Deactivating ONNX profiler')

        try:
            trace_file_path = self._onnx_session.end_profiling()
            trace_file_size = os.path.getsize(trace_file_path)
            if trace_file_size > 50 * 1e6:
                raise Exception('Trace file too big: {0}'.format(trace_file_size))

            with open(trace_file_path) as f:
                trace_json = f.read()
                signal.trace_data = gzip.compress(trace_json.encode())
        finally:
            remove_log_dir(self._log_dir)

