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
import torch.multiprocessing as mp
import deepspeed

import graphsignal
from graphsignal.proto import signals_pb2
from graphsignal.uploader import Uploader
from graphsignal.endpoint_trace import DEFAULT_OPTIONS
from graphsignal.recorders.deepspeed_recorder import DeepSpeedRecorder

logger = logging.getLogger('graphsignal')


class DeepSpeedRecorderTest(unittest.TestCase):
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
        import torch
        if not torch.cuda.is_available() and torch.cuda.device_count() < 2:
            return

        mp.set_start_method("spawn")
        p2 = mp.Process(target=init_process, args=(1, 2))
        p2.start()

        init_process(0, 2)

        recorder = DeepSpeedRecorder()
        recorder.setup()
        signal = signals_pb2.Trace()
        context = {}
        recorder.on_trace_start(signal, context, DEFAULT_OPTIONS)

        tensor = torch.zeros(1)
        tensor += 1
        deepspeed.comm.send(tensor=tensor.cuda(), dst=1)

        recorder.on_trace_stop(signal, context, DEFAULT_OPTIONS)
        recorder.on_trace_read(signal, context, DEFAULT_OPTIONS)

        #pp = pprint.PrettyPrinter()
        #pp.pprint(MessageToJson(signal))

        self.assertEqual(signal.frameworks[0].name, 'DeepSpeed')

        self.assertEqual(signal.frameworks[0].params[0].name, 'deepspeed.comm.get_world_size')
        self.assertEqual(signal.frameworks[0].params[0].value, str(2))

        self.assertEqual(signal.op_profile[0].op_type, signals_pb2.OpStats.OpType.COLLECTIVE_OP)
        self.assertEqual(signal.op_profile[0].op_name, 'send')
        self.assertEqual(signal.op_profile[0].count, 1)
        self.assertTrue(signal.op_profile[0].host_time_ns > 0)
        self.assertTrue(signal.op_profile[0].data_size > 0)
        self.assertTrue(signal.op_profile[0].data_per_sec > 0)

        p2.terminate()


def init_process(rank, size):
    logger.info('Initializing process %d (%d)', rank, size)

    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(size)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29501"

    deepspeed.init_distributed()

    torch.cuda.set_device(rank)

    if rank == 1:
        tensor = torch.zeros(1)
        deepspeed.comm.recv(tensor=tensor.cuda(), src=0)


def start_processes(size):
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()