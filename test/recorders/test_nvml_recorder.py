import unittest
import logging
import sys
from unittest.mock import patch, Mock
import pprint
import time
import socket

import graphsignal
from graphsignal.spans import Span
from graphsignal.uploader import Uploader
from graphsignal.recorders.nvml_recorder import NVMLRecorder

logger = logging.getLogger('graphsignal')

def has_nvidia_gpu() -> bool:
    try:
        import pynvml
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        pynvml.nvmlShutdown()
        return device_count > 0
    except (ImportError, Exception):
        return False

class NVMLRecorderTest(unittest.TestCase):
    def setUp(self):
        if len(logger.handlers) == 0:
            logger.addHandler(logging.StreamHandler(sys.stdout))
        graphsignal.configure(
            api_key='k1',
            debug_mode=True)
        graphsignal._tracer.auto_export = False

    def tearDown(self):
        graphsignal.shutdown()

    def test_record(self):
        if not has_nvidia_gpu():
            self.skipTest("No NVIDIA GPU available")
            return

        recorder = NVMLRecorder()
        recorder.setup()

        import torch
        model = torch.nn.Conv2d(1, 1, kernel_size=(1, 1))
        if torch.cuda.is_available():
            model = model.cuda()

        x = torch.arange(-5, 5, 0.1).view(1, 1, -1, 1)
        if torch.cuda.is_available():
            x = x.cuda()
        pred = model(x)

        recorder.take_snapshot()

        recorder.on_metric_update()

        for device_usage in recorder._last_snapshot:
            print(device_usage)

        tracer = graphsignal._tracer
        if torch.cuda.is_available():
            self.assertTrue(tracer.get_tag('device.bus_id') is not None)
            self.assertTrue(tracer.get_tag('device.uuid') is not None)
            self.assertTrue(tracer.get_tag('device.address') is not None)
            self.assertTrue(tracer.get_tag('device.name') is not None)

        store = tracer.metric_store()
        self.assertTrue(len(store._metrics) > 0)
        metric_tags =  graphsignal._tracer.tags.copy()
        key = store.metric_key('gpu.utilization', metric_tags)
        
        has_gpu_metrics = False
        for key in store._metrics.keys():
            if key[0].startswith('gpu.'):
                has_gpu_metrics = True
                break
        self.assertTrue(has_gpu_metrics)

        self.assertTrue(store._metrics is not None)
        if key in store._metrics:
            self.assertTrue(store._metrics[key].gauge > 0)
        key = store.metric_key('gpu.mxu.utilization', metric_tags)
        if key in store._metrics:
            self.assertTrue(store._metrics[key].gauge > 0)
        key = store.metric_key('gpu.memory.access', metric_tags)
        if key in store._metrics:
            self.assertTrue(store._metrics[key].gauge > 0)
        key = store.metric_key('gpu.memory.usage', metric_tags)
        if key in store._metrics:
            self.assertTrue(store._metrics[key].gauge > 0)
        key = store.metric_key('gpu.temperature', metric_tags)
        if key in store._metrics:
            self.assertTrue(store._metrics[key].gauge > 0)
        key = store.metric_key('gpu.power.usage', metric_tags)
        if key in store._metrics:
            self.assertTrue(store._metrics[key].gauge > 0)
        
        # Test new PCIe metrics
        key = store.metric_key('gpu.pcie.throughput.tx', metric_tags)
        if key in store._metrics:
            self.assertTrue(store._metrics[key].gauge >= 0)
        key = store.metric_key('gpu.pcie.throughput.rx', metric_tags)
        if key in store._metrics:
            self.assertTrue(store._metrics[key].gauge >= 0)
        key = store.metric_key('gpu.pcie.utilization.tx', metric_tags)
        if key in store._metrics:
            self.assertTrue(store._metrics[key].gauge >= 0)
        key = store.metric_key('gpu.pcie.utilization.rx', metric_tags)
        if key in store._metrics:
            self.assertTrue(store._metrics[key].gauge >= 0)
        key = store.metric_key('gpu.pcie.bandwidth.tx', metric_tags)
        if key in store._metrics:
            self.assertTrue(store._metrics[key].gauge >= 0)
        key = store.metric_key('gpu.pcie.bandwidth.rx', metric_tags)
        if key in store._metrics:
            self.assertTrue(store._metrics[key].gauge >= 0)
        key = store.metric_key('gpu.pcie.max_bandwidth', metric_tags)
        if key in store._metrics:
            self.assertTrue(store._metrics[key].gauge > 0)
        
        # Test new NVLINK metrics
        key = store.metric_key('gpu.nvlink.throughput.data.tx', metric_tags)
        if key in store._metrics:
            self.assertTrue(store._metrics[key].gauge >= 0)
        key = store.metric_key('gpu.nvlink.throughput.data.rx', metric_tags)
        if key in store._metrics:
            self.assertTrue(store._metrics[key].gauge >= 0)
        key = store.metric_key('gpu.nvlink.throughput.control.tx', metric_tags)
        if key in store._metrics:
            self.assertTrue(store._metrics[key].gauge >= 0)
        key = store.metric_key('gpu.nvlink.throughput.control.rx', metric_tags)
        if key in store._metrics:
            self.assertTrue(store._metrics[key].gauge >= 0)
        key = store.metric_key('gpu.nvlink.bandwidth.tx', metric_tags)
        if key in store._metrics:
            self.assertTrue(store._metrics[key].gauge >= 0)
        key = store.metric_key('gpu.nvlink.bandwidth.rx', metric_tags)
        if key in store._metrics:
            self.assertTrue(store._metrics[key].gauge >= 0)
        key = store.metric_key('gpu.nvlink.utilization.tx', metric_tags)
        if key in store._metrics:
            self.assertTrue(store._metrics[key].gauge >= 0)
        key = store.metric_key('gpu.nvlink.utilization.rx', metric_tags)
        if key in store._metrics:
            self.assertTrue(store._metrics[key].gauge >= 0)
        key = store.metric_key('gpu.nvlink.link_count', metric_tags)
        if key in store._metrics:
            self.assertTrue(store._metrics[key].gauge >= 0)
        key = store.metric_key('gpu.nvlink.active_links', metric_tags)
        if key in store._metrics:
            self.assertTrue(store._metrics[key].gauge >= 0)
        key = store.metric_key('gpu.nvlink.link_speed', metric_tags)
        if key in store._metrics:
            self.assertTrue(store._metrics[key].gauge >= 0)
        key = store.metric_key('gpu.nvlink.link_width', metric_tags)
        if key in store._metrics:
            self.assertTrue(store._metrics[key].gauge >= 0)
        
        # Test new NVLINK error metrics
        key = store.metric_key('gpu.errors.nvlink.crc', metric_tags)
        if key in store._metrics:
            self.assertTrue(store._metrics[key].gauge >= 0)
        key = store.metric_key('gpu.errors.nvlink.minor', metric_tags)
        if key in store._metrics:
            self.assertTrue(store._metrics[key].gauge >= 0)
        key = store.metric_key('gpu.errors.nvlink.major', metric_tags)
        if key in store._metrics:
            self.assertTrue(store._metrics[key].gauge >= 0)
        key = store.metric_key('gpu.errors.nvlink.fatal', metric_tags)
        if key in store._metrics:
            self.assertTrue(store._metrics[key].gauge >= 0)

    @patch.object(Uploader, 'upload_issue')
    def test_record_xid_errors_mocked(self, mocked_upload_issue):
        if not has_nvidia_gpu():
            self.skipTest("No NVIDIA GPU available")
            return

        recorder = NVMLRecorder()
        recorder.setup()

        recorder.take_snapshot()

        recorder._error_counters[0]['last_xid_error_codes'] = [1, 2, 3]
        recorder.on_metric_update()

        tracer = graphsignal._tracer
        store = tracer.metric_store()
        self.assertTrue(len(store._metrics) > 0)
        metric_tags =  graphsignal._tracer.tags.copy()
        key = store.metric_key('gpu.errors.xid', metric_tags)
        if key in store._metrics:
            self.assertEqual(store._metrics[key].counter, 3)

        issue = mocked_upload_issue.call_args[0][0]

        self.assertEqual(issue.name, 'gpu.errors.xid')
        self.assertEqual(issue.description, 'XID error 3')
