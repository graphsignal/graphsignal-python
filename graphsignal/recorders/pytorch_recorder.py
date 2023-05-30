import logging
import sys
import torch
import torch.distributed as dist

import graphsignal
from graphsignal.recorders.base_recorder import BaseRecorder
from graphsignal.proto_utils import parse_semver
from graphsignal.proto import signals_pb2
from graphsignal.proto_utils import add_library_param, add_driver

logger = logging.getLogger('graphsignal')

class PyTorchRecorder(BaseRecorder):
    def __init__(self):
        self._library = None
        self._comm_info = None
        self._rank = None
        self._is_cuda_available = False

    def setup(self):
        self._library = signals_pb2.LibraryInfo()
        self._library.name = 'PyTorch'
        parse_semver(self._library.version, torch.__version__)

        add_library_param(self._library, 'torch.cuda.is_available', torch.cuda.is_available())
        add_library_param(self._library, 'torch.backends.cuda.is_build', torch.backends.cuda.is_built())
        add_library_param(self._library, 'torch.backends.cudnn.is_available', torch.backends.cudnn.is_available())
        if torch.backends.cudnn.is_available(): 
            try:
                add_library_param(self._library, 'torch.backends.cudnn.is_available', _format_version(torch.backends.cudnn.version()))
            except RuntimeError:
                pass
        if hasattr(torch.backends, 'mps'):
            add_library_param(self._library, 'torch.backends.mps.is_available', torch.backends.mps.is_available())
            add_library_param(self._library, 'torch.backends.mps.is_built', torch.backends.mps.is_built())
        if hasattr(torch.backends, 'mkl'):
            add_library_param(self._library, 'torch.backends.mkl.is_available', torch.backends.mkl.is_available())
        if hasattr(torch.backends, 'mkldnn'):
            add_library_param(self._library, 'torch.backends.mkldnn.is_available', torch.backends.mkldnn.is_available())
        if hasattr(torch.backends, 'openmp'):
            add_library_param(self._library, 'torch.backends.openmp.is_available', torch.backends.openmp.is_available())
        add_library_param(self._library, 'torch.distributed.is_available', torch.distributed.is_available())
        if dist.is_available():
            add_library_param(self._library, 'torch.distributed.is_mpi_available', torch.distributed.is_mpi_available())
            add_library_param(self._library, 'torch.distributed.is_nccl_available', torch.distributed.is_nccl_available())
            add_library_param(self._library, 'torch.distributed.is_initialized', torch.distributed.is_initialized())
            if dist.is_initialized():
                add_library_param(self._library, 'torch.distributed.get_backend', torch.distributed.get_backend())
                add_library_param(self._library, 'torch.distributed.get_world_size', torch.distributed.get_world_size())
                add_library_param(self._library, 'torch.distributed.get_rank', torch.distributed.get_rank())
                self._rank = torch.distributed.get_rank()

        if torch.cuda.is_available():
            self._is_cuda_available = True

    def on_span_start(self, proto, context, options):
        if not options.enable_profiling:
            return
        if self._is_cuda_available:
            context['pytorch_mem_stats'] = {}
            for device in range(torch.cuda.device_count()):
                context['pytorch_mem_stats'][device] = _read_mem_stats(device)

    def on_span_stop(self, proto, context, options):
        if not options.enable_profiling:
            return
        if self._is_cuda_available:
            for device in range(torch.cuda.device_count()):
                if 'pytorch_mem_stats' in context and device in context['pytorch_mem_stats']: 
                    start_mem_stats = context['pytorch_mem_stats'][device]
                    stop_mem_stats = _read_mem_stats(device)

                    mem_diff = _compute_diff(start_mem_stats, stop_mem_stats)
                    mem_alloc = proto.alloc_summary.add()
                    mem_alloc.allocator_type = signals_pb2.MemoryAllocation.AllocatorType.PYTORCH_CUDA_ALLOCATOR
                    mem_alloc.device_idx = device
                    mem_alloc.allocated_size = mem_diff.get('allocated_size', 0)
                    mem_alloc.reserved_size = mem_diff.get('reserved_size', 0)
                    mem_alloc.freed_size = mem_diff.get('freed_size', 0)
                    mem_alloc.num_allocations = mem_diff.get('num_allocations', 0)
                    mem_alloc.num_alloc_retries = mem_diff.get('num_alloc_retries', 0)
                    mem_alloc.num_ooms = mem_diff.get('num_ooms', 0)

    def on_span_read(self, proto, context, options):
        if not options.enable_profiling:
            return
        if self._library:
            proto.libraries.append(self._library)
        if self._rank is not None:
            proto.process_usage.rank = self._rank
            proto.process_usage.has_rank = True


def _format_version(version):
    major = int(version / 1000)
    minor = int(version % 1000 / 100)
    patch = int(version % 10)
    return '{0}.{1}.{2}'.format(major, minor, patch)


def _read_mem_stats(device):
    mem_stats = torch.cuda.memory_stats(device)

    return dict(
        allocated_size=mem_stats.get("allocated_bytes.all.allocated", 0),
        reserved_size=mem_stats.get("allocated_bytes.all.reserved", 0),
        freed_size=mem_stats.get("allocated_bytes.all.freed", 0),
        num_allocations=mem_stats.get("allocation.all.allocated", 0),
        num_alloc_retries=mem_stats.get("num_alloc_retries", 0),
        num_ooms=mem_stats.get("num_ooms", 0)
    )

def _compute_diff(start_mem_stats, stop_mem_stats):
    diff = {}
    for key, stop_value in stop_mem_stats.items():
        start_value = start_mem_stats.get(key, 0)
        change = stop_value - start_value
        if change > 0:
            diff[key] = change
    return diff