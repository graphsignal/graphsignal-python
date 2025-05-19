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
from graphsignal.recorders.vllm_recorder import VLLMRecorder
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

    async def test_llm_generate(self):
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

        print(outputs[0].outputs[0].text)