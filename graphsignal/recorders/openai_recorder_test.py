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
from graphsignal.traces import DEFAULT_OPTIONS
from graphsignal.recorders.openai_recorder import OpenAIRecorder

logger = logging.getLogger('graphsignal')


class OpenAIRecorderTest(unittest.IsolatedAsyncioTestCase):
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
        recorder = OpenAIRecorder()
        recorder.setup()
        proto = signals_pb2.Trace()
        context = {}
        recorder.on_trace_start(proto, context, DEFAULT_OPTIONS)
        recorder.on_trace_stop(proto, context, DEFAULT_OPTIONS)
        recorder.on_trace_read(proto, context, DEFAULT_OPTIONS)
        self.assertEqual(proto.frameworks[0].name, 'OpenAI Python Library')

    @patch.object(Uploader, 'upload_trace')
    @patch.object(openai.Completion, 'create')
    async def test_completion_create(self, mocked_create, mocked_upload_trace):
        # mocking overrides autoinstrumentation, reinstrument
        recorder = OpenAIRecorder()
        recorder.setup()

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

        proto = mocked_upload_trace.call_args[0][0]

        #pp = pprint.PrettyPrinter()
        #pp.pprint(MessageToJson(proto))

        self.assertEqual(find_tag(proto, 'component'), 'LLM')
        self.assertEqual(find_tag(proto, 'operation'), 'openai.Completion.create')
        self.assertEqual(find_tag(proto, 'endpoint'), 'https://api.openai.com/v1/completions')

        self.assertEqual(find_param(proto, 'model'), 'text-davinci-003')
        self.assertEqual(find_param(proto, 'max_tokens'), '1024')
        self.assertEqual(find_param(proto, 'temperature'), '0.1')
        self.assertEqual(find_param(proto, 'top_p'), '1')
        self.assertEqual(find_param(proto, 'presence_penalty'), '0')
        self.assertEqual(find_param(proto, 'frequency_penalty'), '0')

        self.assertEqual(find_data_count(proto, 'prompt', 'byte_count'), 26.0)
        self.assertEqual(find_data_count(proto, 'prompt', 'element_count'), 2.0)
        self.assertEqual(find_data_count(proto, 'prompt', 'token_count'), 6.0)
        self.assertEqual(find_data_count(proto, 'completion', 'byte_count'), 256.0)
        self.assertEqual(find_data_count(proto, 'completion', 'element_count'), 15.0)
        self.assertEqual(find_data_count(proto, 'completion', 'token_count'), 96.0)
        self.assertEqual(find_data_count(proto, 'completion', 'finish_reason_stop'), 1.0)
        self.assertEqual(find_data_count(proto, 'completion', 'finish_reason_length'), 1.0)

    @patch.object(Uploader, 'upload_trace')
    @patch.object(openai.Completion, 'create')
    async def test_completion_create_stream(self, mocked_create, mocked_upload_trace):
        # mocking overrides autoinstrumentation, reinstrument
        recorder = OpenAIRecorder()
        recorder.setup()

        test_ret = [
            {
                "choices": [
                    {
                    "finish_reason": None,
                    "index": 1,
                    "logprobs": None,
                    "text": "abc"
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
                    "finish_reason": "stop",
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

        proto = mocked_upload_trace.call_args[0][0]

        #pp = pprint.PrettyPrinter()
        #pp.pprint(MessageToJson(proto))
        self.assertEqual(find_data_count(proto, 'prompt', 'byte_count'), 37.0)
        self.assertEqual(find_data_count(proto, 'prompt', 'element_count'), 2.0)
        self.assertEqual(find_data_count(proto, 'completion', 'byte_count'), 4.0)
        self.assertEqual(find_data_count(proto, 'completion', 'element_count'), 2.0)
        self.assertEqual(find_data_count(proto, 'completion', 'finish_reason_stop'), 1.0)

    @patch.object(Uploader, 'upload_trace')
    @patch.object(openai.Completion, 'acreate')
    async def test_completion_acreate(self, mocked_create, mocked_upload_trace):
        # mocking overrides autoinstrumentation, reinstrument
        recorder = OpenAIRecorder()
        recorder.setup()

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

        proto = mocked_upload_trace.call_args[0][0]

        #pp = pprint.PrettyPrinter()
        #pp.pprint(MessageToJson(proto))

        self.assertEqual(find_tag(proto, 'component'), 'LLM')
        self.assertEqual(find_tag(proto, 'operation'), 'openai.Completion.acreate')
        self.assertEqual(find_tag(proto, 'endpoint'), 'https://api.openai.com/v1/completions')

        self.assertEqual(find_param(proto, 'model'), 'text-davinci-003')
        self.assertEqual(find_param(proto, 'max_tokens'), '1024')
        self.assertEqual(find_param(proto, 'temperature'), '0.1')
        self.assertEqual(find_param(proto, 'top_p'), '1')
        self.assertEqual(find_param(proto, 'presence_penalty'), '0')
        self.assertEqual(find_param(proto, 'frequency_penalty'), '0')

        self.assertEqual(find_data_count(proto, 'prompt', 'byte_count'), 26.0)
        self.assertEqual(find_data_count(proto, 'prompt', 'element_count'), 2.0)
        self.assertEqual(find_data_count(proto, 'prompt', 'token_count'), 6.0)
        self.assertEqual(find_data_count(proto, 'completion', 'byte_count'), 254.0)
        self.assertEqual(find_data_count(proto, 'completion', 'element_count'), 15.0)
        self.assertEqual(find_data_count(proto, 'completion', 'token_count'), 96.0)

    @patch.object(Uploader, 'upload_trace')
    @patch.object(openai.ChatCompletion, 'create')
    async def test_chat_completion_create(self, mocked_create, mocked_upload_trace):
        # mocking overrides autoinstrumentation, reinstrument
        recorder = OpenAIRecorder()
        recorder.setup()

        mocked_create.return_value = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "logprobs": None,
                    "message": {
                        "content": "\n\ntest completion 1"
                    }
                },
                {
                    "finish_reason": "length",
                    "index": 1,
                    "logprobs": None,
                    "message": {
                        "content": "\n\ntest completion 222"
                    }
                }
            ],
            "created": 1675070122,
            "id": "cmpl-id",
            "model": "gpt-3.5-turbo",
            "object": "text_completion",
            "usage": {
                "completion_tokens": 96,
                "prompt_tokens": 6,
                "total_tokens": 102
            }
        }

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", 
            messages=[
                {'role': 'system', 'content': 'test prompt 1'}, 
                {'role': 'system', 'content': 'test prompt 2'}],
            temperature=0.1,
            top_p=1,
            max_tokens=1024,
            frequency_penalty=0,
            presence_penalty=0)

        recorder.shutdown()

        proto = mocked_upload_trace.call_args[0][0]

        #pp = pprint.PrettyPrinter()
        #pp.pprint(MessageToJson(proto))

        self.assertEqual(find_tag(proto, 'component'), 'LLM')
        self.assertEqual(find_tag(proto, 'operation'), 'openai.ChatCompletion.create')
        self.assertEqual(find_tag(proto, 'endpoint'), 'https://api.openai.com/v1/chat/completions')

        self.assertEqual(find_param(proto, 'model'), 'gpt-3.5-turbo')
        self.assertEqual(find_param(proto, 'max_tokens'), '1024')
        self.assertEqual(find_param(proto, 'temperature'), '0.1')
        self.assertEqual(find_param(proto, 'top_p'), '1')
        self.assertEqual(find_param(proto, 'presence_penalty'), '0')
        self.assertEqual(find_param(proto, 'frequency_penalty'), '0')

        self.assertEqual(find_data_count(proto, 'messages', 'byte_count'), 38.0)
        self.assertEqual(find_data_count(proto, 'messages', 'element_count'), 4.0)
        self.assertEqual(find_data_count(proto, 'messages', 'token_count'), 6.0)
        self.assertEqual(find_data_count(proto, 'completion', 'byte_count'), 253.0)
        self.assertEqual(find_data_count(proto, 'completion', 'element_count'), 15.0)
        self.assertEqual(find_data_count(proto, 'completion', 'token_count'), 96.0)
        self.assertEqual(find_data_count(proto, 'completion', 'finish_reason_stop'), 1.0)
        self.assertEqual(find_data_count(proto, 'completion', 'finish_reason_length'), 1.0)

    @patch.object(Uploader, 'upload_trace')
    @patch.object(openai.ChatCompletion, 'create')
    async def test_chat_completion_create_stream(self, mocked_create, mocked_upload_trace):
        # mocking overrides autoinstrumentation, reinstrument
        recorder = OpenAIRecorder()
        recorder.setup()

        test_ret = [
            {
                "choices": [
                    {
                        "finish_reason": None,
                        "index": 1,
                        "logprobs": None,
                        "delta": {
                            "content": "abc"
                        }
                    }
                ],
                "created": 1676896808,
                "id": "cmpl-6lzkernKqOvF4ewZGhHlZ63HZJcbc",
                "model": "gpt-3.5-turbo",
                "object": "chat.completion.chunk"
            },
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "index": 1,
                        "logprobs": None,
                        "delta": {
                            "content": "\n"
                        }
                    }
                ],
                "created": 1676896808,
                "id": "cmpl-6lzkernKqOvF4ewZGhHlZ63HZJcbc",
                "model": "gpt-3.5-turbo",
                "object": "chat.completion.chunk"
            }
        ]
        def test_ret_gen():
            for item in test_ret:
                yield item
        mocked_create.return_value = test_ret_gen()

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", 
            messages=[
                {'role': 'system', 'content': 'count 1 to 3'}, 
                {'role': 'system', 'content': 'generate 2 random letters'}],            
            temperature=0.1,
            top_p=1,
            max_tokens=1024,
            frequency_penalty=0,
            presence_penalty=0,
            stream=True)

        for r in response:
            pass

        recorder.shutdown()

        proto = mocked_upload_trace.call_args[0][0]

        #pp = pprint.PrettyPrinter()
        #pp.pprint(MessageToJson(proto))
        self.assertEqual(find_data_count(proto, 'messages', 'byte_count'), 49.0)
        self.assertEqual(find_data_count(proto, 'messages', 'element_count'), 4.0)
        self.assertEqual(find_data_count(proto, 'completion', 'byte_count'), 4.0)
        self.assertEqual(find_data_count(proto, 'completion', 'element_count'), 2.0)
        self.assertEqual(find_data_count(proto, 'completion', 'finish_reason_stop'), 1.0)

    @patch.object(Uploader, 'upload_trace')
    @patch.object(openai.Edit, 'create')
    async def test_edits_create(self, mocked_create, mocked_upload_trace):
        # mocking overrides autoinstrumentation, reinstrument
        recorder = OpenAIRecorder()
        recorder.setup()

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

        proto = mocked_upload_trace.call_args[0][0]

        #pp = pprint.PrettyPrinter()
        #pp.pprint(MessageToJson(proto))

        self.assertEqual(find_tag(proto, 'component'), 'LLM')
        self.assertEqual(find_tag(proto, 'operation'), 'openai.Edit.create')
        self.assertEqual(find_tag(proto, 'endpoint'), 'https://api.openai.com/v1/edits')

        self.assertEqual(find_param(proto, 'model'), 'text-davinci-edit-001')
        self.assertEqual(find_param(proto, 'temperature'), '0.1')
        self.assertEqual(find_param(proto, 'top_p'), '1')

        self.assertEqual(find_data_count(proto, 'input', 'byte_count'), 12.0)
        self.assertEqual(find_data_count(proto, 'input', 'element_count'), 1.0)
        self.assertEqual(find_data_count(proto, 'instruction', 'byte_count'), 18.0)
        self.assertEqual(find_data_count(proto, 'instruction', 'element_count'), 1.0)
        self.assertEqual(find_data_count(proto, 'instruction', 'token_count'), 18.0)
        self.assertEqual(find_data_count(proto, 'edits', 'byte_count'), 32.0)
        self.assertEqual(find_data_count(proto, 'edits', 'element_count'), 1.0)
        self.assertEqual(find_data_count(proto, 'edits', 'token_count'), 13.0)

    @patch.object(Uploader, 'upload_trace')
    @patch.object(openai.Embedding, 'create')
    async def test_embedding_create(self, mocked_create, mocked_upload_trace):
        # mocking overrides autoinstrumentation, reinstrument
        recorder = OpenAIRecorder()
        recorder.setup()

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

        proto = mocked_upload_trace.call_args[0][0]

        #pp = pprint.PrettyPrinter()
        #pp.pprint(MessageToJson(proto))

        self.assertEqual(find_tag(proto, 'component'), 'LLM')
        self.assertEqual(find_tag(proto, 'operation'), 'openai.Embedding.create')
        self.assertEqual(find_tag(proto, 'endpoint'), 'https://api.openai.com/v1/embeddings')

        self.assertEqual(find_param(proto, 'engine'), 'text-embedding-ada-002')

        self.assertEqual(find_data_count(proto, 'input', 'byte_count'), 22.0)
        self.assertEqual(find_data_count(proto, 'input', 'element_count'), 2.0)
        self.assertEqual(find_data_count(proto, 'input', 'token_count'), 8.0)
        self.assertEqual(find_data_count(proto, 'embedding', 'byte_count'), 144.0)
        self.assertEqual(find_data_count(proto, 'embedding', 'element_count'), 6.0)

    @patch.object(Uploader, 'upload_trace')
    @patch.object(openai.Image, 'create')
    async def test_image_create(self, mocked_create, mocked_upload_trace):
        # mocking overrides autoinstrumentation, reinstrument
        recorder = OpenAIRecorder()
        recorder.setup()

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

        proto = mocked_upload_trace.call_args[0][0]

        #pp = pprint.PrettyPrinter()
        #pp.pprint(MessageToJson(proto))

        self.assertEqual(find_tag(proto, 'component'), 'Model')
        self.assertEqual(find_tag(proto, 'operation'), 'openai.Image.create')
        self.assertEqual(find_tag(proto, 'endpoint'), 'https://api.openai.com/v1/images/generations')

        self.assertEqual(find_param(proto, 'n'), '1')
        self.assertEqual(find_param(proto, 'size'), '256x256')
        self.assertEqual(find_param(proto, 'response_format'), 'b64_json')

        self.assertEqual(find_data_count(proto, 'image', 'byte_count'), 14.0)
        self.assertEqual(find_data_count(proto, 'image', 'element_count'), 1.0)

    @patch.object(Uploader, 'upload_trace')
    @patch.object(openai.Audio, 'transcribe')
    async def test_audio_transcription(self, mocked_create, mocked_upload_trace):
        # mocking overrides autoinstrumentation, reinstrument
        recorder = OpenAIRecorder()
        recorder.setup()

        mocked_create.return_value = {
            "text": 'some text'
        }

        import tempfile
        with tempfile.NamedTemporaryFile() as tmp_file:
            tmp_file.write('test content'.encode('utf-8'))
            tmp_file.flush()

            response = openai.Audio.transcribe(
                file=tmp_file,
                model='whisper-1',
                response_format='json',
                prompt='test prompt',
                temperature=0.1,
                language='en')

        recorder.shutdown()

        proto = mocked_upload_trace.call_args[0][0]

        #pp = pprint.PrettyPrinter()
        #pp.pprint(MessageToJson(proto))

        self.assertEqual(find_tag(proto, 'component'), 'Model')
        self.assertEqual(find_tag(proto, 'operation'), 'openai.Audio.transcribe')
        self.assertEqual(find_tag(proto, 'endpoint'), 'https://api.openai.com/v1/audio/transcriptions')

        self.assertEqual(find_param(proto, 'model'), 'whisper-1')
        self.assertEqual(find_param(proto, 'response_format'), 'json')
        self.assertEqual(find_param(proto, 'temperature'), '0.1')
        self.assertEqual(find_param(proto, 'language'), 'en')

        self.assertEqual(find_data_count(proto, 'file', 'byte_count'), 12.0)
        self.assertEqual(find_data_count(proto, 'prompt', 'byte_count'), 11.0)
        self.assertEqual(find_data_count(proto, 'prompt', 'element_count'), 1.0)


    @patch.object(Uploader, 'upload_trace')
    @patch.object(openai.Audio, 'translate')
    async def test_audio_translate(self, mocked_create, mocked_upload_trace):
        # mocking overrides autoinstrumentation, reinstrument
        recorder = OpenAIRecorder()
        recorder.setup()

        mocked_create.return_value = {
            "text": 'some text'
        }

        import tempfile
        with tempfile.NamedTemporaryFile() as tmp_file:
            tmp_file.write('test content'.encode('utf-8'))
            tmp_file.flush()

            response = openai.Audio.translate(
                file=tmp_file,
                model='whisper-1',
                response_format='json',
                prompt='test prompt',
                temperature=0.1)

        recorder.shutdown()

        proto = mocked_upload_trace.call_args[0][0]

        #pp = pprint.PrettyPrinter()
        #pp.pprint(MessageToJson(proto))

        self.assertEqual(find_tag(proto, 'component'), 'Model')
        self.assertEqual(find_tag(proto, 'operation'), 'openai.Audio.translate')
        self.assertEqual(find_tag(proto, 'endpoint'), 'https://api.openai.com/v1/audio/translations')

        self.assertEqual(find_param(proto, 'model'), 'whisper-1')
        self.assertEqual(find_param(proto, 'response_format'), 'json')
        self.assertEqual(find_param(proto, 'temperature'), '0.1')

        self.assertEqual(find_data_count(proto, 'file', 'byte_count'), 12.0)
        self.assertEqual(find_data_count(proto, 'prompt', 'byte_count'), 11.0)
        self.assertEqual(find_data_count(proto, 'prompt', 'element_count'), 1.0)

    @patch.object(Uploader, 'upload_trace')
    @patch.object(openai.Moderation, 'create')
    async def test_moderation_create(self, mocked_create, mocked_upload_trace):
        # mocking overrides autoinstrumentation, reinstrument
        recorder = OpenAIRecorder()
        recorder.setup()

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

        proto = mocked_upload_trace.call_args[0][0]

        #pp = pprint.PrettyPrinter()
        #pp.pprint(MessageToJson(proto))

        self.assertEqual(find_tag(proto, 'component'), 'Model')
        self.assertEqual(find_tag(proto, 'operation'), 'openai.Moderation.create')
        self.assertEqual(find_tag(proto, 'endpoint'), 'https://api.openai.com/v1/moderations')

        self.assertEqual(find_param(proto, 'model'), 'text-moderation-latest')

        self.assertEqual(find_data_count(proto, 'input', 'byte_count'), 9.0)
        self.assertEqual(find_data_count(proto, 'input', 'element_count'), 1.0)


def find_tag(proto, key):
    for tag in proto.tags:
        if tag.key == key:
            return tag.value
    return None


def find_param(proto, name):
    for param in proto.params:
        if param.name == name:
            return param.value
    return None


def find_data_count(proto, data_name, count_name):
    for data_stats in proto.data_profile:
        if data_stats.data_name == data_name:
            for data_count in data_stats.counts:
                if data_count.name == count_name:
                    return data_count.count
    return None