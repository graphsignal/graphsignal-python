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
from graphsignal import Tracer
from graphsignal import spans
from graphsignal.recorders.pytorch_recorder import PyTorchRecorder
from test.model_utils import find_tag, find_attribute

logger = logging.getLogger('graphsignal')


class PytorchRecorderTest(unittest.TestCase):
    def setUp(self):
        if len(logger.handlers) == 0:
            logger.addHandler(logging.StreamHandler(sys.stdout))
        graphsignal.configure(
            api_key='k1',
            debug_mode=True)
        graphsignal._tracer.auto_export = False

    def tearDown(self):
        graphsignal.shutdown()

    @patch.object(Tracer, 'emit_span_start')
    @patch.object(Tracer, 'emit_span_stop')
    @patch.object(Tracer, 'emit_span_read')
    def test_record(self, mock_emit_span_start, mock_emit_span_stop, mock_emit_span_read):

        recorder = PyTorchRecorder()
        recorder.setup()

        x = torch.arange(-5, 5, 0.1).view(-1, 1)
        model = torch.nn.Linear(1, 1)
        if torch.cuda.is_available():
            x = x.to('cuda')
            model = model.to('cuda')

        span = spans.Span('op1')
        context = {}
        recorder.on_span_start(span, context)

        pred = model(x)

        recorder.on_span_stop(span, context)
        recorder.on_span_read(span, context)

        #pp = pprint.PrettyPrinter()
        #pp.pprint(span._profiles['profile.pytorch.cpu'].content)

        tracer = graphsignal._tracer
        self.assertEqual(tracer.get_tag('framework.name'), 'pytorch')
        self.assertEqual(tracer.get_tag('framework.version'), torch.__version__)
        
        prof = span._profiles['profile.pytorch.cpu']
        cpu_events = json.loads(prof.content)
        self.assertEqual(prof.name, 'profile.pytorch.cpu')
        self.assertEqual(prof.format, 'event-averages')

        test_event = None
        for event in cpu_events:
            if event['op_name'] == 'aten::addmm':
                test_event = event
                break
        self.assertIsNotNone(test_event)
        self.assertTrue(test_event['count'] >= 1)
        
        if torch.cuda.is_available():
            self.assertTrue(test_event['device_time_ns'] >= 1)
            self.assertTrue(test_event['self_device_time_ns'] >= 1)
        else:
            self.assertTrue(test_event['cpu_time_ns'] >= 1)
            self.assertTrue(test_event['self_cpu_time_ns'] >= 1)

        if torch.cuda.is_available():
            prof = span._profiles['profile.pytorch.kernel']
            device_events = json.loads(prof.content)
            self.assertEqual(prof.name, 'profile.pytorch.kernel')
            self.assertEqual(prof.format, 'event-averages')
            # todo: check kernel events

        prof = span._profiles['profile.pytorch.trace']
        json.loads(prof.content)
        self.assertEqual(prof.name, 'profile.pytorch.trace')
        self.assertEqual(prof.format, 'chrome-trace')
        self.assertTrue('aten::addmm' in prof.content)

    def test_on_metric_update(self):
        """Test that PyTorch memory metrics are recorded correctly"""
        if not torch.cuda.is_available():
            self.skipTest("No CUDA available")
            return

        recorder = PyTorchRecorder()
        recorder.setup()

        # Create some GPU memory usage
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()

        # Call the metric update method
        recorder.on_metric_update()

        # Verify that metrics were recorded
        tracer = graphsignal._tracer
        store = tracer.metric_store()
        self.assertTrue(len(store._metrics) > 0)

        # Check for PyTorch memory metrics
        has_pytorch_metrics = False
        for key in store._metrics.keys():
            if key[0].startswith('pytorch.memory.'):
                has_pytorch_metrics = True
                break
        self.assertTrue(has_pytorch_metrics)

        # Test specific memory metrics
        metric_tags = graphsignal._tracer.tags.copy()
        metric_tags.update({'device.index': 0, 'device.type': 'gpu', 'framework.name': 'pytorch'})

        # Basic memory metrics
        key = store.metric_key('pytorch.memory.allocated', metric_tags)
        if key in store._metrics:
            self.assertTrue(store._metrics[key].gauge >= 0)

        key = store.metric_key('pytorch.memory.reserved', metric_tags)
        if key in store._metrics:
            self.assertTrue(store._metrics[key].gauge >= 0)

        key = store.metric_key('pytorch.memory.total', metric_tags)
        if key in store._metrics:
            self.assertTrue(store._metrics[key].gauge > 0)

        key = store.metric_key('pytorch.memory.utilization', metric_tags)
        if key in store._metrics:
            self.assertTrue(store._metrics[key].gauge >= 0)

        # Memory management metrics
        key = store.metric_key('pytorch.memory.alloc_retries', metric_tags)
        if key in store._metrics:
            self.assertTrue(store._metrics[key].gauge >= 0)

        key = store.metric_key('pytorch.memory.ooms', metric_tags)
        if key in store._metrics:
            self.assertTrue(store._metrics[key].gauge >= 0)

        key = store.metric_key('pytorch.memory.sync_all_streams', metric_tags)
        if key in store._metrics:
            self.assertTrue(store._metrics[key].gauge >= 0)

        key = store.metric_key('pytorch.memory.device_alloc', metric_tags)
        if key in store._metrics:
            self.assertTrue(store._metrics[key].gauge >= 0)

        key = store.metric_key('pytorch.memory.device_free', metric_tags)
        if key in store._metrics:
            self.assertTrue(store._metrics[key].gauge >= 0)

        # Clean up
        del x, y
        torch.cuda.empty_cache()
        