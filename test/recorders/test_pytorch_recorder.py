import unittest
import logging
import sys
import os
import json
import time
from unittest.mock import patch, Mock
from google.protobuf.json_format import MessageToJson
import pprint
import torch

import graphsignal
from graphsignal.proto import signals_pb2
from graphsignal.uploader import Uploader
from graphsignal.spans import DEFAULT_OPTIONS
from graphsignal.recorders.pytorch_recorder import PyTorchRecorder

logger = logging.getLogger('graphsignal')


class PyTorchRecorderTest(unittest.TestCase):
    def setUp(self):
        if len(logger.handlers) == 0:
            logger.addHandler(logging.StreamHandler(sys.stdout))
        graphsignal.configure(
            api_key='k1',
            deployment='d1',
            upload_on_shutdown=False,
            debug_mode=True)

    def tearDown(self):
        graphsignal.shutdown()

    def test_record(self):
        recorder = PyTorchRecorder()
        recorder.setup()

        mem1 = torch.rand(10000)
        if torch.cuda.is_available():
            mem1 = mem1.cuda()
            torch.cuda.synchronize()

        proto = signals_pb2.Span()
        context = {}
        recorder.on_span_start(proto, context, graphsignal.TraceOptions(enable_profiling=True))

        mem2 = torch.rand(10000)
        if torch.cuda.is_available():
            mem2 = mem2.cuda()
        del mem2
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        recorder.on_span_stop(proto, context, graphsignal.TraceOptions(enable_profiling=True))
        recorder.on_span_read(proto, context, graphsignal.TraceOptions(enable_profiling=True))

        self.assertEqual(proto.libraries[0].name, 'PyTorch')

        self.assertEqual(proto.libraries[0].params[0].name, 'torch.cuda.is_available')
        self.assertEqual(proto.libraries[0].params[0].value, str(torch.cuda.is_available()))

        if torch.cuda.is_available():
            self.assertEqual(proto.alloc_summary[0].allocator_type, signals_pb2.MemoryAllocation.AllocatorType.PYTORCH_CUDA_ALLOCATOR)
            self.assertEqual(proto.alloc_summary[0].device_idx, 0)
            self.assertTrue(proto.alloc_summary[0].allocated_size > 0)
            self.assertTrue(proto.alloc_summary[0].freed_size > 0)
            self.assertEqual(proto.alloc_summary[0].num_allocations, 1)
