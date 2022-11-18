import logging
import sys
import torch
import torch.distributed as dist

import graphsignal
from graphsignal.recorders.base_recorder import BaseRecorder
from graphsignal.proto_utils import parse_semver
from graphsignal.proto import signals_pb2
from graphsignal.proto_utils import add_framework_param, add_driver

logger = logging.getLogger('graphsignal')

class PyTorchRecorder(BaseRecorder):
    def __init__(self):
        self._framework = None
        self._comm_info = None

    def setup(self):
        self._framework = signals_pb2.FrameworkInfo()
        self._framework.type = signals_pb2.FrameworkInfo.FrameworkType.PYTORCH_FRAMEWORK
        parse_semver(self._framework.version, torch.__version__)

        add_framework_param(self._framework, 'torch.cuda.is_available', torch.cuda.is_available())
        add_framework_param(self._framework, 'torch.backends.cuda.is_build', torch.backends.cuda.is_built())
        add_framework_param(self._framework, 'torch.backends.cudnn.is_available', torch.backends.cudnn.is_available())
        if torch.backends.cudnn.is_available(): 
            try:
                add_framework_param(self._framework, 'torch.backends.cudnn.is_available', _format_version(torch.backends.cudnn.version()))
            except RuntimeError:
                pass
        if hasattr(torch.backends, 'mps'):
            add_framework_param(self._framework, 'torch.backends.mps.is_available', torch.backends.mps.is_available())
            add_framework_param(self._framework, 'torch.backends.mps.is_built', torch.backends.mps.is_built())
        if hasattr(torch.backends, 'mkl'):
            add_framework_param(self._framework, 'torch.backends.mkl.is_available', torch.backends.mkl.is_available())
        if hasattr(torch.backends, 'mkldnn'):
            add_framework_param(self._framework, 'torch.backends.mkldnn.is_available', torch.backends.mkldnn.is_available())
        if hasattr(torch.backends, 'openmp'):
            add_framework_param(self._framework, 'torch.backends.openmp.is_available', torch.backends.openmp.is_available())
        add_framework_param(self._framework, 'torch.distributed.is_available', torch.distributed.is_available())
        if dist.is_available():
            add_framework_param(self._framework, 'torch.distributed.is_mpi_available', torch.distributed.is_mpi_available())
            add_framework_param(self._framework, 'torch.distributed.is_nccl_available', torch.distributed.is_nccl_available())
            add_framework_param(self._framework, 'torch.distributed.is_initialized', torch.distributed.is_initialized())
            if dist.is_initialized():
                add_framework_param(self._framework, 'torch.distributed.get_backend', torch.distributed.get_backend())
                add_framework_param(self._framework, 'torch.distributed.get_world_size', torch.distributed.get_world_size())
                add_framework_param(self._framework, 'torch.distributed.get_rank', torch.distributed.get_rank())

    def on_trace_start(self, signal, context):
        pass

    def on_trace_stop(self, signal, context):
        if self._framework:
            signal.frameworks.append(self._framework)


def _format_version(version):
    major = int(version / 1000)
    minor = int(version % 1000 / 100)
    patch = int(version % 10)
    return '{0}.{1}.{2}'.format(major, minor, patch)
