import unittest
import logging
import sys
import os
import json
import time
from unittest.mock import patch, Mock
from google.protobuf.json_format import MessageToJson
import pprint
import openai
import transformers
from transformers.tools import OpenAiAgent

import graphsignal
from graphsignal.proto import signals_pb2
from graphsignal.uploader import Uploader
from graphsignal.spans import DEFAULT_OPTIONS
from graphsignal.recorders.openai_recorder import OpenAIRecorder
from graphsignal.recorders.huggingface_recorder import HuggingFaceRecorder
from graphsignal.proto_utils import find_tag, find_param, find_data_sample

logger = logging.getLogger('graphsignal')


class HuggingFaceRecorderTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        if len(logger.handlers) == 0:
            logger.addHandler(logging.StreamHandler(sys.stdout))
        graphsignal.configure(
            api_key='k1',
            deployment='d1',
            upload_on_shutdown=False,
            debug_mode=True)

    async def asyncTearDown(self):
        graphsignal.shutdown()

    async def test_record(self):
        recorder = HuggingFaceRecorder()
        recorder.setup()
        proto = signals_pb2.Span()
        context = {}
        recorder.on_span_start(proto, context, DEFAULT_OPTIONS)
        recorder.on_span_stop(proto, context, DEFAULT_OPTIONS)
        recorder.on_span_read(proto, context, DEFAULT_OPTIONS)

        self.assertEqual(proto.libraries[0].name, 'Transformers')

    @patch.object(Uploader, 'upload_span')
    @patch.object(openai.Completion, 'create')
    @patch.object(transformers.tools.agents, 'get_remote_tools')
    @patch.object(transformers.tools.RemoteTool, '__call__')
    async def test_trace_run(self, mocked_remote_tool, mocked_get_remote_tools, mocked_create, mocked_upload_span):
        now = int(time.time())

        transformers.tools.agents.HUGGINGFACE_DEFAULT_TOOLS_FROM_HUB = []
        from transformers.tools.agents import PreTool
        mocked_get_remote_tools.return_value = {
            'translator': PreTool(
                task='translation', 
                description="This is a tool that translates text from a language to another. It takes three inputs: `text`, which should be the text to translate, `src_lang`, which should be the language of the text to translate and `tgt_lang`, which should be the language for the desired ouput language. Both `src_lang` and `tgt_lang` are written in plain English, such as 'Romanian', or 'Albanian'. It returns the text translated in `tgt_lang`.", repo_id=None)
        }

        mocked_remote_tool.return_value = 'Bonjour, comment allez-vous?'

        mocked_create.return_value = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "logprobs": None,
                    "text": " tool: `translator` to translate the text to French.\n\nAnswer:\n```py\ntranslated_text = translator(text=\"Hi, how are you?\", src_lang=\"English\", tgt_lang=\"French\")\nprint(f\"The translated text is {translated_text}.\")\n```"
                },
            ],
            "created": now,
            "id": "cmpl-id",
            "model": "text-davinci-003",
            "object": "text_completion",
            "usage": {
                "completion_tokens": 69,
                "prompt_tokens": 1595,
                "total_tokens": 1664
            }
        }

        # mocking overrides autoinstrumentation, reinstrument
        oai_rec = OpenAIRecorder()
        oai_rec.setup()
        hf_rec = HuggingFaceRecorder()
        hf_rec.setup()

        os.environ['OPENAI_API_KEY'] = 'fake-key'
        agent = OpenAiAgent(model="text-davinci-003")
        agent.run('Translate the following text to French: System update available.', remote=True)

        hf_rec.shutdown()
        oai_rec.shutdown()

        s3 = mocked_upload_span.call_args_list[0][0][0]
        s2 = mocked_upload_span.call_args_list[1][0][0]
        s1 = mocked_upload_span.call_args_list[2][0][0]

        self.assertEqual(find_tag(s1, 'component'), 'Agent')
        self.assertEqual(find_tag(s1, 'operation'), 'transformers.tools.Agent.run')
        self.assertIsNotNone(find_data_sample(s1, 'task'))
        self.assertIsNotNone(find_data_sample(s1, 'output'))

        self.assertEqual(find_tag(s2, 'component'), 'Tool')
        self.assertEqual(find_tag(s2, 'operation'), 'transformers.tools.TranslationTool')
        self.assertIsNotNone(find_tag(s2, 'endpoint'))
        self.assertIsNotNone(find_data_sample(s2, 'inputs'))
        self.assertIsNotNone(find_data_sample(s2, 'outputs'))

        self.assertEqual(find_tag(s3, 'component'), 'LLM')
