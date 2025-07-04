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
from test.model_utils import find_tag, find_param, find_counter

logger = logging.getLogger('graphsignal')


class VLLMRecorderTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.maxDiff = None
        if len(logger.handlers) == 0:
            logger.addHandler(logging.StreamHandler(sys.stdout))
        graphsignal.configure(
            api_key='k1',
            debug_mode=True)
        graphsignal._tracer.export_on_shutdown = False

    async def asyncTearDown(self):
        graphsignal.shutdown()

    @patch.object(Uploader, 'upload_span')
    async def test_llm_generate(self, mocked_upload_span):
        if not torch.cuda.is_available():
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

        model = mocked_upload_span.call_args[0][0]

        self.assertEqual(find_tag(model, 'model'), 'gpt2')
        self.assertEqual(find_param(model, 'model'), 'gpt2')

        self.assertTrue(find_counter(model, 'latency_ns') > 0)
        #self.assertEqual(find_counter(model, 'output_tokens'), 18)
        #self.assertEqual(find_counter(model, 'prompt_tokens'), 78)
        #self.assertEqual(find_counter(model, 'completion_tokens'), 18)

        #print("Model output:")
        #print(outputs[0].outputs[0].text)