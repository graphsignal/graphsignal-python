import unittest
import logging
import sys
import os
import json
import time
from unittest.mock import patch, Mock
from google.protobuf.json_format import MessageToJson
import pprint
import types
import openai

import graphsignal
from graphsignal.proto import signals_pb2
from graphsignal.uploader import Uploader
from graphsignal.endpoint_trace import DEFAULT_OPTIONS
from graphsignal.recorders.openai_recorder import OpenAIRecorder

logger = logging.getLogger('graphsignal')


class OpenAIRecorderTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        if len(logger.handlers) == 0:
            logger.addHandler(logging.StreamHandler(sys.stdout))
        graphsignal.configure(
            api_key='k1',
            deployment='d1',
            debug_mode=True)

    async def asyncTearDown(self):
        graphsignal.shutdown()

    async def test_record(self):
        recorder = OpenAIRecorder()
        recorder.setup()
        signal = signals_pb2.WorkerSignal()
        context = {}
        recorder.on_trace_start(signal, context, DEFAULT_OPTIONS)
        recorder.on_trace_stop(signal, context, DEFAULT_OPTIONS)
        recorder.on_trace_read(signal, context, DEFAULT_OPTIONS)

        self.assertEqual(signal.frameworks[0].name, 'OpenAI Python Library')

    @patch.object(Uploader, 'upload_signal')
    @patch.object(openai.Completion, 'create')
    async def test_completion_create(self, mocked_create, mocked_upload_signal):
        # mocking overrides autoinstrumentation, reinstrument
        recorder = OpenAIRecorder()
        recorder.setup()
        recorder._is_sampling = True

        mocked_create.return_value = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "logprobs": None,
                    "text": "\n\ntest completion 1"
                },
                {
                    "finish_reason": "length",
                    "index": 1,
                    "logprobs": None,
                    "text": "\n\ntest completion 222"
                }
            ],
            "created": 1675070122,
            "id": "cmpl-id",
            "model": "text-davinci-003",
            "object": "text_completion",
            "usage": {
                "completion_tokens": 96,
                "prompt_tokens": 6,
                "total_tokens": 102
            }
        }

        response = openai.Completion.create(
            model="text-davinci-003", 
            prompt=['test prompt 1', 'test prompt 2'],
            temperature=0.1,
            top_p=1,
            max_tokens=1024,
            frequency_penalty=0,
            presence_penalty=0)

        recorder.shutdown()

        signal = mocked_upload_signal.call_args[0][0]

        #pp = pprint.PrettyPrinter()
        #pp.pprint(MessageToJson(signal))

        self.assertEqual(signal.frameworks[0].name, 'OpenAI Python Library')

        self.assertEqual(find_param(signal, 'model'), 'text-davinci-003')
        self.assertEqual(find_param(signal, 'max_tokens'), '1024')
        self.assertEqual(find_param(signal, 'temperature'), '0.1')
        self.assertEqual(find_param(signal, 'top_p'), '1')
        self.assertEqual(find_param(signal, 'presence_penalty'), '0')
        self.assertEqual(find_param(signal, 'frequency_penalty'), '0')

        self.assertEqual(find_data_metric(signal, 'prompt', 'byte_count'), 26.0)
        self.assertEqual(find_data_metric(signal, 'prompt', 'element_count'), 2.0)
        self.assertEqual(find_data_metric(signal, 'prompt', 'token_count'), 6.0)
        self.assertEqual(find_data_metric(signal, 'completion', 'byte_count'), 40.0)
        self.assertEqual(find_data_metric(signal, 'completion', 'element_count'), 2.0)
        self.assertEqual(find_data_metric(signal, 'completion', 'token_count'), 96.0)
        self.assertEqual(find_data_metric(signal, 'completion', 'finish_reason_stop'), 1.0)
        self.assertEqual(find_data_metric(signal, 'completion', 'finish_reason_length'), 1.0)

    @patch.object(Uploader, 'upload_signal')
    @patch.object(openai.Completion, 'create')
    async def test_completion_create_stream(self, mocked_create, mocked_upload_signal):
        # mocking overrides autoinstrumentation, reinstrument
        recorder = OpenAIRecorder()
        recorder.setup()
        recorder._is_sampling = True

        test_ret = [
            {
                "choices": [
                    {
                    "finish_reason": None,
                    "index": 1,
                    "logprobs": None,
                    "text": "\n"
                    }
                ],
                "created": 1676896808,
                "id": "cmpl-6lzkernKqOvF4ewZGhHlZ63HZJcbc",
                "model": "text-davinci-003",
                "object": "text_completion"
            },
            {
                "choices": [
                    {
                    "finish_reason": None,
                    "index": 1,
                    "logprobs": None,
                    "text": "\n"
                    }
                ],
                "created": 1676896808,
                "id": "cmpl-6lzkernKqOvF4ewZGhHlZ63HZJcbc",
                "model": "text-davinci-003",
                "object": "text_completion"
            }
        ]
        def test_ret_gen():
            for item in test_ret:
                yield item
        mocked_create.return_value = test_ret_gen()

        response = openai.Completion.create(
            model="text-davinci-003", 
            prompt=['count 1 to 3', 'generate 2 random letters'],
            temperature=0.1,
            top_p=1,
            max_tokens=1024,
            frequency_penalty=0,
            presence_penalty=0,
            stream=True)

        for r in response:
            pass

        recorder.shutdown()

        signal = mocked_upload_signal.call_args[0][0]

        #pp = pprint.PrettyPrinter()
        #pp.pprint(MessageToJson(signal))

        self.assertEqual(signal.frameworks[0].name, 'OpenAI Python Library')
        self.assertEqual(signal.root_span.spans[0].name, 'response')
        self.assertTrue(signal.root_span.spans[0].start_ns > 0)
        self.assertTrue(signal.root_span.spans[0].end_ns > 0)

    @patch.object(Uploader, 'upload_signal')
    @patch.object(openai.Completion, 'acreate')
    async def test_completion_acreate(self, mocked_create, mocked_upload_signal):
        # mocking overrides autoinstrumentation, reinstrument
        recorder = OpenAIRecorder()
        recorder.setup()
        recorder._is_sampling = True

        mocked_create.return_value = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "logprobs": None,
                    "text": "\n\ntest completion 1"
                },
                {
                    "finish_reason": "stop",
                    "index": 1,
                    "logprobs": None,
                    "text": "\n\ntest completion 222"
                }
            ],
            "created": 1675070122,
            "id": "cmpl-id",
            "model": "text-davinci-003",
            "object": "text_completion",
            "usage": {
                "completion_tokens": 96,
                "prompt_tokens": 6,
                "total_tokens": 102
            }
        }

        response = await openai.Completion.acreate(
            model="text-davinci-003", 
            prompt=['test prompt 1', 'test prompt 2'],
            temperature=0.1,
            top_p=1,
            max_tokens=1024,
            frequency_penalty=0,
            presence_penalty=0)

        recorder.shutdown()

        signal = mocked_upload_signal.call_args[0][0]

        #pp = pprint.PrettyPrinter()
        #pp.pprint(MessageToJson(signal))

        self.assertEqual(signal.frameworks[0].name, 'OpenAI Python Library')

        self.assertEqual(find_param(signal, 'model'), 'text-davinci-003')
        self.assertEqual(find_param(signal, 'max_tokens'), '1024')
        self.assertEqual(find_param(signal, 'temperature'), '0.1')
        self.assertEqual(find_param(signal, 'top_p'), '1')
        self.assertEqual(find_param(signal, 'presence_penalty'), '0')
        self.assertEqual(find_param(signal, 'frequency_penalty'), '0')

        self.assertEqual(find_data_metric(signal, 'prompt', 'byte_count'), 26.0)
        self.assertEqual(find_data_metric(signal, 'prompt', 'element_count'), 2.0)
        self.assertEqual(find_data_metric(signal, 'prompt', 'token_count'), 6.0)
        self.assertEqual(find_data_metric(signal, 'completion', 'byte_count'), 40.0)
        self.assertEqual(find_data_metric(signal, 'completion', 'element_count'), 2.0)
        self.assertEqual(find_data_metric(signal, 'completion', 'token_count'), 96.0)

    @patch.object(Uploader, 'upload_signal')
    @patch.object(openai.Edit, 'create')
    async def test_edits_create(self, mocked_create, mocked_upload_signal):
        # mocking overrides autoinstrumentation, reinstrument
        recorder = OpenAIRecorder()
        recorder.setup()
        recorder._is_sampling = True

        mocked_create.return_value = {
            "choices": [
                {
                    "index": 0,
                    "text": "test input 1\ntest instruction 1\n"
                }
            ],
            "created": 1675077592,
            "object": "edit",
            "usage": {
                "completion_tokens": 13,
                "prompt_tokens": 18,
                "total_tokens": 32
            }
        }

        response = openai.Edit.create(
            model="text-davinci-edit-001", 
            input='test input 1',
            instruction='test instruction 1',
            temperature=0.1,
            top_p=1)

        recorder.shutdown()

        signal = mocked_upload_signal.call_args[0][0]

        #pp = pprint.PrettyPrinter()
        #pp.pprint(MessageToJson(signal))

        self.assertEqual(signal.frameworks[0].name, 'OpenAI Python Library')

        self.assertEqual(find_param(signal, 'model'), 'text-davinci-edit-001')
        self.assertEqual(find_param(signal, 'temperature'), '0.1')
        self.assertEqual(find_param(signal, 'top_p'), '1')

        self.assertEqual(find_data_metric(signal, 'input', 'byte_count'), 12.0)
        self.assertEqual(find_data_metric(signal, 'input', 'element_count'), 1.0)
        self.assertEqual(find_data_metric(signal, 'instruction', 'byte_count'), 18.0)
        self.assertEqual(find_data_metric(signal, 'instruction', 'element_count'), 1.0)
        self.assertEqual(find_data_metric(signal, 'instruction', 'token_count'), 18.0)
        self.assertEqual(find_data_metric(signal, 'edits', 'byte_count'), 32.0)
        self.assertEqual(find_data_metric(signal, 'edits', 'element_count'), 1.0)
        self.assertEqual(find_data_metric(signal, 'edits', 'token_count'), 13.0)

    @patch.object(Uploader, 'upload_signal')
    @patch.object(openai.Embedding, 'create')
    async def test_embedding_create(self, mocked_create, mocked_upload_signal):
        # mocking overrides autoinstrumentation, reinstrument
        recorder = OpenAIRecorder()
        recorder.setup()
        recorder._is_sampling = True

        mocked_create.return_value = {
            "data": [
                {
                    "embedding": [
                        0.0007817253354005516,
                        -0.004567846190184355,
                        -0.011692041531205177
                    ],
                    "index": 0,
                    "object": "embedding"
                    },
                    {
                    "embedding": [
                        -0.008782976306974888,
                        -0.0015969815431162715,
                        -0.004394866060465574
                    ],
                    "index": 1,
                    "object": "embedding"
                    }
                ],
                "model": "text-embedding-ada-002-v2",
                "object": "list",
                "usage": {
                    "prompt_tokens": 8,
                    "total_tokens": 8
                }
            }

        response = openai.Embedding.create(
            engine="text-embedding-ada-002", 
            input=['test text 1', 'test text 2'])

        recorder.shutdown()

        signal = mocked_upload_signal.call_args[0][0]

        #pp = pprint.PrettyPrinter()
        #pp.pprint(MessageToJson(signal))

        self.assertEqual(signal.frameworks[0].name, 'OpenAI Python Library')

        self.assertEqual(find_param(signal, 'engine'), 'text-embedding-ada-002')

        self.assertEqual(find_data_metric(signal, 'input', 'byte_count'), 22.0)
        self.assertEqual(find_data_metric(signal, 'input', 'element_count'), 2.0)
        self.assertEqual(find_data_metric(signal, 'input', 'token_count'), 8.0)
        self.assertEqual(find_data_metric(signal, 'embedding', 'byte_count'), 144.0)
        self.assertEqual(find_data_metric(signal, 'embedding', 'element_count'), 6.0)

    @patch.object(Uploader, 'upload_signal')
    @patch.object(openai.Image, 'create')
    async def test_image_create(self, mocked_create, mocked_upload_signal):
        # mocking overrides autoinstrumentation, reinstrument
        recorder = OpenAIRecorder()
        recorder.setup()
        recorder._is_sampling = True

        mocked_create.return_value = {
            "created": 1675079434,
            "data": [
                {
                    "b64_json": "iVBORw0KGgoAAA"
                }
            ]
        }

        response = openai.Image.create(
            prompt='test image',
            n=1,
            size='256x256',
            response_format='b64_json')

        recorder.shutdown()

        signal = mocked_upload_signal.call_args[0][0]

        #pp = pprint.PrettyPrinter()
        #pp.pprint(MessageToJson(signal))

        self.assertEqual(signal.frameworks[0].name, 'OpenAI Python Library')

        self.assertEqual(find_param(signal, 'n'), '1')
        self.assertEqual(find_param(signal, 'size'), '256x256')
        self.assertEqual(find_param(signal, 'response_format'), 'b64_json')

        self.assertEqual(find_data_metric(signal, 'image', 'byte_count'), 14.0)
        self.assertEqual(find_data_metric(signal, 'image', 'element_count'), 1.0)

    @patch.object(Uploader, 'upload_signal')
    @patch.object(openai.Moderation, 'create')
    async def test_moderation_create(self, mocked_create, mocked_upload_signal):
        # mocking overrides autoinstrumentation, reinstrument
        recorder = OpenAIRecorder()
        recorder.setup()
        recorder._is_sampling = True

        mocked_create.return_value = {
            "id": "modr-id",
            "model": "text-moderation-004",
            "results": [
                {
                    "categories": {
                        "hate": False,
                        "hate/threatening": False,
                        "self-harm": False,
                        "sexual": False,
                        "sexual/minors": False,
                        "violence": False,
                        "violence/graphic": False
                    },
                    "category_scores": {
                        "hate": 1.5125541722227354e-05,
                        "hate/threatening": 4.6567969036459544e-08,
                        "self-harm": 4.265221065224978e-09,
                        "sexual": 0.00031956686871126294,
                        "sexual/minors": 4.4396051634976175e-06,
                        "violence": 6.315908080978261e-07,
                        "violence/graphic": 6.493833097920287e-07
                    },
                    "flagged": False
                }
            ]
        }

        response = openai.Moderation.create(
            model='text-moderation-latest',
            input='test text')

        recorder.shutdown()

        signal = mocked_upload_signal.call_args[0][0]

        #pp = pprint.PrettyPrinter()
        #pp.pprint(MessageToJson(signal))

        self.assertEqual(signal.frameworks[0].name, 'OpenAI Python Library')

        self.assertEqual(find_param(signal, 'model'), 'text-moderation-latest')

        self.assertEqual(find_data_metric(signal, 'input', 'byte_count'), 9.0)
        self.assertEqual(find_data_metric(signal, 'input', 'element_count'), 1.0)


def find_param(signal, name):
    for param in signal.params:
        if param.name == name:
            return param.value
    return None


def find_data_metric(signal, data_name, metric_name):
    for data_metric in signal.data_metrics:
        if data_metric.data_name == data_name and data_metric.metric_name == metric_name:
            return data_metric.metric.gauge
    return None