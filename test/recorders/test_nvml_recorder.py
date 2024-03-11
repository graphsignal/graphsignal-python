import unittest
import logging
import sys
from unittest.mock import patch, Mock
import torch
from google.protobuf.json_format import MessageToJson
import pprint
import time
import socket

import graphsignal
from graphsignal.recorders.nvml_recorder import NVMLRecorder

logger = logging.getLogger('graphsignal')


class NVMLRecorderTest(unittest.TestCase):
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
        recorder = NVMLRecorder()
        recorder.setup()

        model = torch.nn.Conv2d(1, 1, kernel_size=(1, 1))
        if torch.cuda.is_available():
            model = model.cuda()

        x = torch.arange(-5, 5, 0.1).view(1, 1, -1, 1)
        if torch.cuda.is_available():
            x = x.cuda()
        pred = model(x)

        recorder.take_snapshot()

        recorder.on_metric_update()

        store = graphsignal._tracer.metric_store()
        if len(store._metrics) > 0:
            metric_tags =  {'deployment': 'd1', 'hostname': socket.gethostname(), 'device': 0}
            key = store.metric_key('system', 'gpu_utilization', metric_tags)
            if key in store._metrics:
                self.assertTrue(store._metrics[key].gauge > 0)
            key = store.metric_key('system', 'mxu_utilization', metric_tags)
            if key in store._metrics:
                self.assertTrue(store._metrics[key].gauge > 0)
            key = store.metric_key('system', 'device_memory_access', metric_tags)
            if key in store._metrics:
                self.assertTrue(store._metrics[key].gauge > 0)
            key = store.metric_key('system', 'device_memory_used', metric_tags)
            self.assertTrue(store._metrics[key].gauge > 0)
            key = store.metric_key('system', 'nvlink_throughput_data_tx_kibs', metric_tags)
            if key in store._metrics:
                self.assertTrue(store._metrics[key].gauge > 0)
            key = store.metric_key('system', 'nvlink_throughput_data_rx_kibs', metric_tags)
            if key in store._metrics:
                self.assertTrue(store._metrics[key].gauge > 0)
            key = store.metric_key('system', 'gpu_temp_c', metric_tags)
            self.assertTrue(store._metrics[key].gauge > 0)
            key = store.metric_key('system', 'power_usage_w', metric_tags)
            self.assertTrue(store._metrics[key].gauge > 0)
