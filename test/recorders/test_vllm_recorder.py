import unittest
import logging
import sys
import os
import json
import time
import base64
from unittest.mock import patch, Mock
import pprint
import types
import torch

import graphsignal
from graphsignal import client
from graphsignal.uploader import Uploader
from graphsignal.tracer import Tracer
from test.model_utils import find_tag, find_attribute, find_counter

logger = logging.getLogger('graphsignal')


class VLLMRecorderTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.maxDiff = None
        if len(logger.handlers) == 0:
            logger.addHandler(logging.StreamHandler(sys.stdout))
        graphsignal.configure(
            api_key='k1',
            debug_mode=True)
        graphsignal._tracer.auto_export = False

    async def asyncTearDown(self):
        graphsignal.shutdown()

    @patch.object(Uploader, 'upload_span')
    async def test_llm_generate(self, mocked_upload_span):
        try:
            import vllm
        except ImportError:
            self.skipTest("vllm is not installed")
            return

        if not torch.cuda.is_available():
            self.skipTest("No CUDA available")
            return

        from vllm import LLM, SamplingParams

        llm = LLM(
            model="gpt2",
            enforce_eager=True
        )

        sampling_params = SamplingParams(
            temperature=0.7, 
            top_p=0.95, 
            max_tokens=256
        )

        outputs = llm.generate([f"What is 2 raised to 10 power?"], sampling_params)
        outputs = llm.generate([f"What is 2 raised to 10 power?"], sampling_params)

        print("Model output:")
        print(outputs[0].outputs[0].text)

        span = mocked_upload_span.call_args[0][0]

        self.assertEqual(span.name, 'vllm.llm.generate')
        self.assertEqual(find_tag(span, 'inference.engine.name'), 'vllm')
        self.assertEqual(find_tag(span, 'inference.engine.version'), vllm.__version__)
        self.assertEqual(find_tag(span, 'vllm.model.name'), 'gpt2')
        self.assertEqual(find_attribute(span, 'vllm.model.name'), 'gpt2')

        self.assertTrue(find_counter(span, 'span.duration') > 0)
        #self.assertEqual(find_counter(model, 'output_tokens'), 18)
        #self.assertEqual(find_counter(model, 'prompt_tokens'), 78)
        #self.assertEqual(find_counter(model, 'completion_tokens'), 18)

    @patch.object(Uploader, 'upload_span')
    @patch.object(Tracer, 'should_sample', return_value=True)
    def test_convert_otel_span(self, mocked_should_sample, mocked_upload_span):
        try:
            import vllm
        except ImportError:
            self.skipTest("vllm is not installed")
            return
            
        from graphsignal.recorders.vllm_recorder import VLLMRecorder
        
        recorder = VLLMRecorder()
        
        # Create a mock OTEL span with all possible attributes
        mock_otel_span = Mock()
        mock_otel_span.name = "llm_request"
        mock_otel_span.start_time = 1000000000  # 1 second in nanoseconds
        mock_otel_span.end_time = 2000000000    # 2 seconds in nanoseconds
        mock_otel_span.attributes = {
            'gen_ai.request.id': 'test_request_123',
            'gen_ai.response.model': 'Qwen/Qwen1.5-7B-Chat',
            'gen_ai.request.temperature': 0.7,
            'gen_ai.request.top_p': 0.95,
            'gen_ai.request.max_tokens': 256,
            'gen_ai.request.n': 1,
            'gen_ai.usage.num_sequences': 1,
            'gen_ai.usage.prompt_tokens': 16,
            'gen_ai.usage.completion_tokens': 1,
            'gen_ai.latency.time_in_queue': 0.012926340103149414,
            'gen_ai.latency.time_to_first_token': 0.22484421730041504,
            'gen_ai.latency.e2e': 0.22525858879089355,
            'gen_ai.latency.time_in_scheduler': 0.0013919062912464142
        }
        
        recorder._convert_otel_span(mock_otel_span)
        
        # Verify upload_span was called
        mocked_upload_span.assert_called_once()
        span = mocked_upload_span.call_args[0][0]

        # Check basic span properties
        self.assertEqual(span.start_ns, 1000000000)
        self.assertEqual(span.end_ns, 2000000000)
        self.assertEqual(span.name, 'vllm.llm_request')

        self.assertEqual(find_attribute(span, 'sampling.reason'), 'vllm.otel')
        
        # Check request attributes
        self.assertEqual(find_tag(span, 'vllm.request.id'), 'test_request_123')
        
        # Check model attributes
        self.assertEqual(find_tag(span, 'vllm.response.model'), 'Qwen/Qwen1.5-7B-Chat')
        self.assertEqual(find_attribute(span, 'vllm.response.model'), 'Qwen/Qwen1.5-7B-Chat')
        
        # Check request parameters
        self.assertEqual(find_attribute(span, 'vllm.request.temperature'), '0.7')
        self.assertEqual(find_attribute(span, 'vllm.request.top_p'), '0.95')
        self.assertEqual(find_attribute(span, 'vllm.request.max_tokens'), '256')
        self.assertEqual(find_attribute(span, 'vllm.request.n'), '1')
        
        # Check usage metrics
        self.assertEqual(find_counter(span, 'vllm.usage.num_sequences'), 1.0)
        self.assertEqual(find_counter(span, 'vllm.usage.prompt_tokens'), 16.0)
        self.assertEqual(find_counter(span, 'vllm.usage.completion_tokens'), 1.0)
        
        # Check latency metrics
        self.assertEqual(find_counter(span, 'vllm.latency.time_in_queue'), 0.012926340103149414)
        self.assertEqual(find_counter(span, 'vllm.latency.time_to_first_token'), 0.22484421730041504)
        self.assertEqual(find_counter(span, 'vllm.latency.e2e'), 0.22525858879089355)
        self.assertEqual(find_counter(span, 'vllm.latency.time_in_scheduler'), 0.0013919062912464142)

