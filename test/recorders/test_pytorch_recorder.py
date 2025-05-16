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

logger = logging.getLogger('graphsignal')


class PytorchRecorderTest(unittest.TestCase):
    def setUp(self):
        if len(logger.handlers) == 0:
            logger.addHandler(logging.StreamHandler(sys.stdout))
        graphsignal.configure(
            api_key='k1',
            profiling_rate=1,
            debug_mode=True)
        graphsignal._tracer.export_on_shutdown = False

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
            x = x.to('cuda:0')
            model = model.to('cuda:0')

        span = spans.Span('op1')
        context = {}
        recorder.on_span_start(span, context)

        pred = model(x)

        recorder.on_span_stop(span, context)
        recorder.on_span_read(span, context)

        #pp = pprint.PrettyPrinter()
        #pp.pprint(span._profiles['operations'].content)

        self.assertEqual(span.get_param('profiler'), f'pytorch-{torch.__version__}')
        
        prof = span._profiles['pytorch-cpu-profile']
        cpu_events = json.loads(prof.content)
        self.assertEqual(prof.name, 'pytorch-cpu-profile')
        self.assertEqual(prof.format, 'event-averages')

        test_event = None
        for event in cpu_events:
            if event['op_name'] == 'aten::addmm':
                test_event = event
                break
        self.assertIsNotNone(test_event)
        self.assertTrue(test_event['count'] >= 1)
        
        if torch.cuda.is_available():
            self.assertEqual(test_event['device_type'], 'CUDA')
            self.assertTrue(test_event['device_time_ns'] >= 1)
            self.assertTrue(test_event['self_device_time_ns'] >= 1)
        else:
            self.assertEqual(test_event['device_type'], 'CPU')
            self.assertTrue(test_event['cpu_time_ns'] >= 1)
            self.assertTrue(test_event['self_cpu_time_ns'] >= 1)

        if torch.cuda.is_available():
            prof = span._profiles['pytorch-kernel-profile']
            device_events = json.loads(prof.content)
            self.assertEqual(prof.name, 'pytorch-kernel-profile')
            self.assertEqual(prof.format, 'event-averages')
            # todo: check kernel events

        prof = span._profiles['pytorch-timeline']
        json.loads(prof.content)
        self.assertEqual(prof.name, 'pytorch-timeline')
        self.assertEqual(prof.format, 'chrome-trace')
        self.assertTrue('aten::addmm' in prof.content)
        