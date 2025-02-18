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
from graphsignal import spans
from graphsignal.recorders.pytorch_recorder import PyTorchRecorder

logger = logging.getLogger('graphsignal')


class KinetoRecorderTest(unittest.TestCase):
    def setUp(self):
        if len(logger.handlers) == 0:
            logger.addHandler(logging.StreamHandler(sys.stdout))
        graphsignal.configure(
            api_key='k1',
            deployment='d1',
            upload_on_shutdown=False,
            profiling_rate=1,
            debug_mode=True)

    def tearDown(self):
        graphsignal.shutdown()

    def test_record(self):
        recorder = PyTorchRecorder()
        recorder.setup()

        x = torch.arange(-5, 5, 0.1).view(-1, 1)
        model = torch.nn.Linear(1, 1)
        if torch.cuda.is_available():
            x = x.to('cuda:0')
            model = model.to('cuda:0')

        span = spans.Span('op1', with_profile=True)
        context = {}
        recorder.on_span_start(span, context)

        pred = model(x)

        recorder.on_span_stop(span, context)
        recorder.on_span_read(span, context)

        #pp = pprint.PrettyPrinter()
        #pp.pprint(span._profiles['operations'].content)

        if torch.cuda.is_available():
            self.assertEqual(span.get_tag('profile_type'), 'device')
        else:
            self.assertEqual(span.get_tag('profile_type'), 'cpu')
        self.assertEqual(span.get_tag('profiler'), f'pytorch-{torch.__version__}')
        
        cpu_profile = span._profiles['cpu-profile']
        cpu_events = json.loads(cpu_profile.content)
        self.assertEqual(cpu_profile.name, 'cpu-profile')
        self.assertEqual(cpu_profile.format, 'event-averages')

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
            device_profile = span._profiles['device-profile']
            device_events = json.loads(device_profile.content)
            self.assertEqual(device_profile.name, 'device-profile')
            self.assertEqual(device_profile.format, 'event-averages')
            # todo: check kernel events
