import unittest
import logging
import sys
from unittest.mock import patch, Mock
import pprint
import time
import socket

import graphsignal
from graphsignal.signals.spans import Span
from graphsignal.recorders.nvml_recorder import NVMLRecorder
from graphsignal.proto import signals_pb2
from test.test_utils import find_last_datapoint

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
        graphsignal._ticker.auto_tick = False

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
        _ = model(x)

        recorder.take_snapshot()

        recorder.on_tick()

        ticker = graphsignal._ticker
        if torch.cuda.is_available():
            self.assertTrue(ticker.get_tag('device.bus_id') is not None)
            self.assertTrue(ticker.get_tag('device.uuid') is not None)
            self.assertTrue(ticker.get_tag('device.address') is not None)
            self.assertTrue(ticker.get_tag('device.name') is not None)

        store = ticker.metric_store()
        self.assertTrue(len(store._metrics) > 0)
        metric_tags =  graphsignal._ticker.tags.copy()
        key = store.metric_key('gpu.utilization', metric_tags)
        
        has_gpu_metrics = False
        for key in store._metrics.keys():
            if key[0].startswith('gpu.'):
                has_gpu_metrics = True
                break
        self.assertTrue(has_gpu_metrics)

        self.assertTrue(store._metrics is not None)
        if key in store._metrics:
            self.assertTrue(find_last_datapoint(store, key).gauge > 0)
        key = store.metric_key('gpu.mxu.utilization', metric_tags)
        if key in store._metrics:
            self.assertTrue(find_last_datapoint(store, key).gauge > 0)
        key = store.metric_key('gpu.memory.access', metric_tags)
        if key in store._metrics:
            self.assertTrue(find_last_datapoint(store, key).gauge > 0)
        key = store.metric_key('gpu.memory.usage', metric_tags)
        if key in store._metrics:
            self.assertTrue(find_last_datapoint(store, key).gauge > 0)
        key = store.metric_key('gpu.temperature', metric_tags)
        if key in store._metrics:
            self.assertTrue(find_last_datapoint(store, key).gauge > 0)
        key = store.metric_key('gpu.power.usage', metric_tags)
        if key in store._metrics:
            self.assertTrue(find_last_datapoint(store, key).gauge > 0)
        
        # Test new PCIe metrics
        key = store.metric_key('gpu.pcie.throughput.tx', metric_tags)
        if key in store._metrics:
            self.assertTrue(find_last_datapoint(store, key).gauge >= 0)
        key = store.metric_key('gpu.pcie.throughput.rx', metric_tags)
        if key in store._metrics:
            self.assertTrue(find_last_datapoint(store, key).gauge >= 0)
        key = store.metric_key('gpu.pcie.utilization.tx', metric_tags)
        if key in store._metrics:
            self.assertTrue(find_last_datapoint(store, key).gauge >= 0)
        key = store.metric_key('gpu.pcie.utilization.rx', metric_tags)
        if key in store._metrics:
            self.assertTrue(find_last_datapoint(store, key).gauge >= 0)
        key = store.metric_key('gpu.pcie.bandwidth.tx', metric_tags)
        if key in store._metrics:
            self.assertTrue(find_last_datapoint(store, key).gauge >= 0)
        key = store.metric_key('gpu.pcie.bandwidth.rx', metric_tags)
        if key in store._metrics:
            self.assertTrue(find_last_datapoint(store, key).gauge >= 0)
        key = store.metric_key('gpu.pcie.max_bandwidth', metric_tags)
        if key in store._metrics:
            self.assertTrue(find_last_datapoint(store, key).gauge > 0)
        
        # Test new NVLINK metrics
        key = store.metric_key('gpu.nvlink.throughput.data.tx', metric_tags)
        if key in store._metrics:
            self.assertTrue(find_last_datapoint(store, key).gauge >= 0)
        key = store.metric_key('gpu.nvlink.throughput.data.rx', metric_tags)
        if key in store._metrics:
            self.assertTrue(find_last_datapoint(store, key).gauge >= 0)
        key = store.metric_key('gpu.nvlink.throughput.control.tx', metric_tags)
        if key in store._metrics:
            self.assertTrue(find_last_datapoint(store, key).gauge >= 0)
        key = store.metric_key('gpu.nvlink.throughput.control.rx', metric_tags)
        if key in store._metrics:
            self.assertTrue(find_last_datapoint(store, key).gauge >= 0)
        key = store.metric_key('gpu.nvlink.bandwidth.tx', metric_tags)
        if key in store._metrics:
            self.assertTrue(find_last_datapoint(store, key).gauge >= 0)
        key = store.metric_key('gpu.nvlink.bandwidth.rx', metric_tags)
        if key in store._metrics:
            self.assertTrue(find_last_datapoint(store, key).gauge >= 0)
        key = store.metric_key('gpu.nvlink.utilization.tx', metric_tags)
        if key in store._metrics:
            self.assertTrue(find_last_datapoint(store, key).gauge >= 0)
        key = store.metric_key('gpu.nvlink.utilization.rx', metric_tags)
        if key in store._metrics:
            self.assertTrue(find_last_datapoint(store, key).gauge >= 0)
        key = store.metric_key('gpu.nvlink.link_count', metric_tags)
        if key in store._metrics:
            self.assertTrue(find_last_datapoint(store, key).gauge >= 0)
        key = store.metric_key('gpu.nvlink.active_links', metric_tags)
        if key in store._metrics:
            self.assertTrue(find_last_datapoint(store, key).gauge >= 0)
        key = store.metric_key('gpu.nvlink.link_speed', metric_tags)
        if key in store._metrics:
            self.assertTrue(find_last_datapoint(store, key).gauge >= 0)
        key = store.metric_key('gpu.nvlink.link_width', metric_tags)
        if key in store._metrics:
            self.assertTrue(find_last_datapoint(store, key).gauge >= 0)
        
        # Test new NVLINK error metrics
        key = store.metric_key('gpu.errors.nvlink.crc', metric_tags)
        if key in store._metrics:
            self.assertTrue(find_last_datapoint(store, key).gauge >= 0)
        key = store.metric_key('gpu.errors.nvlink.minor', metric_tags)
        if key in store._metrics:
            self.assertTrue(find_last_datapoint(store, key).gauge >= 0)
        key = store.metric_key('gpu.errors.nvlink.major', metric_tags)
        if key in store._metrics:
            self.assertTrue(find_last_datapoint(store, key).gauge >= 0)
        key = store.metric_key('gpu.errors.nvlink.fatal', metric_tags)
        if key in store._metrics:
            self.assertTrue(find_last_datapoint(store, key).gauge >= 0)

    def test_record_xid_errors_mocked(self):
        if not has_nvidia_gpu():
            self.skipTest("No NVIDIA GPU available")
            return

        recorder = NVMLRecorder()
        recorder.setup()

        recorder.take_snapshot()

        # Clear log store before test
        ticker = graphsignal._ticker
        log_store = ticker.log_store()
        log_store.clear()
        
        recorder._error_counters[0]['last_xid_error_codes'] = [1, 2, 3]
        recorder.on_tick()

        store = ticker.metric_store()
        self.assertTrue(len(store._metrics) > 0)
        metric_tags =  graphsignal._ticker.tags.copy()
        key = store.metric_key('gpu.errors.xid', metric_tags)
        if key in store._metrics:
            self.assertEqual(find_last_datapoint(store, key).total, 3)

        # Check log messages instead of errors
        log_batches = log_store.export()
        
        # Find log entries for XID errors
        xid_log_entries = []
        for batch in log_batches:
            for entry in batch.log_entries:
                if 'XID error' in entry.message:
                    xid_log_entries.append(entry)
        
        # Should have 3 log entries for the 3 XID errors
        self.assertEqual(len(xid_log_entries), 3)
        
        # Check that all error codes are present
        error_codes = set()
        for entry in xid_log_entries:
            self.assertEqual(entry.level, signals_pb2.LogEntry.LogLevel.ERROR_LEVEL)
            # Extract error code from message like "XID error 1"
            if 'XID error' in entry.message:
                error_code = entry.message.split('XID error')[1].strip()
                error_codes.add(int(error_code))
        
        self.assertEqual(error_codes, {1, 2, 3})
