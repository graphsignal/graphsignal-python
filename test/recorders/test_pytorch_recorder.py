import unittest
import logging
import sys
import os
import json
import time
from unittest.mock import patch, Mock
import pprint
import torch

import graphsignal
from graphsignal.core.ticker import Ticker
from graphsignal.signals.spans import Span
from graphsignal.recorders.pytorch_recorder import PyTorchRecorder
from test.test_utils import find_tag, find_attribute, find_last_datapoint

logger = logging.getLogger('graphsignal')


class PytorchRecorderTest(unittest.TestCase):
    def setUp(self):
        if len(logger.handlers) == 0:
            logger.addHandler(logging.StreamHandler(sys.stdout))
        graphsignal.configure(
            api_key='k1',
            debug_mode=True)
        graphsignal._ticker.auto_tick = False

    def tearDown(self):
        graphsignal.shutdown()

    def test_on_tick(self):
        if not torch.cuda.is_available():
            self.skipTest("No CUDA available")
            return

        recorder = PyTorchRecorder()
        recorder.setup()

        # Create some GPU memory usage
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()

        # Call the metric update method
        recorder.on_tick()

        # Verify that metrics were recorded
        ticker = graphsignal._ticker
        store = ticker.metric_store()
        self.assertTrue(len(store._metrics) > 0)

        # Check for PyTorch memory metrics
        has_pytorch_metrics = False
        for key in store._metrics.keys():
            if key[0].startswith('pytorch.memory.'):
                has_pytorch_metrics = True
                break
        self.assertTrue(has_pytorch_metrics)

        # Test specific memory metrics
        metric_tags = graphsignal._ticker.tags.copy()
        metric_tags.update({'device.index': 0, 'device.type': 'gpu', 'framework.name': 'pytorch'})

        # Basic memory metrics
        key = store.metric_key('pytorch.memory.allocated', metric_tags)
        if key in store._metrics:
            self.assertTrue(find_last_datapoint(store, key).gauge >= 0)

        key = store.metric_key('pytorch.memory.reserved', metric_tags)
        if key in store._metrics:
            self.assertTrue(find_last_datapoint(store, key).gauge >= 0)

        key = store.metric_key('pytorch.memory.total', metric_tags)
        if key in store._metrics:
            self.assertTrue(find_last_datapoint(store, key).gauge > 0)

        key = store.metric_key('pytorch.memory.utilization', metric_tags)
        if key in store._metrics:
            self.assertTrue(find_last_datapoint(store, key).gauge >= 0)

        # Memory management metrics
        key = store.metric_key('pytorch.memory.alloc_retries', metric_tags)
        if key in store._metrics:
            self.assertTrue(find_last_datapoint(store, key).gauge >= 0)

        key = store.metric_key('pytorch.memory.ooms', metric_tags)
        if key in store._metrics:
            self.assertTrue(find_last_datapoint(store, key).gauge >= 0)

        key = store.metric_key('pytorch.memory.sync_all_streams', metric_tags)
        if key in store._metrics:
            self.assertTrue(find_last_datapoint(store, key).gauge >= 0)

        key = store.metric_key('pytorch.memory.device_alloc', metric_tags)
        if key in store._metrics:
            self.assertTrue(find_last_datapoint(store, key).gauge >= 0)

        key = store.metric_key('pytorch.memory.device_free', metric_tags)
        if key in store._metrics:
            self.assertTrue(find_last_datapoint(store, key).gauge >= 0)

        # Clean up
        del x, y
        torch.cuda.empty_cache()
        