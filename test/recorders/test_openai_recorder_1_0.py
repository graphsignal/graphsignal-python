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
from graphsignal.spans import DEFAULT_OPTIONS
from graphsignal.recorders.openai_recorder import OpenAIRecorder
from graphsignal.proto_utils import find_tag, find_param, find_data_count, find_data_sample
from graphsignal.proto_utils import parse_semver, compare_semver
from graphsignal.proto import signals_pb2

logger = logging.getLogger('graphsignal')

openai_lib = signals_pb2.LibraryInfo()
parse_semver(openai_lib.version, openai.version.VERSION)

if compare_semver(openai_lib.version, (1, 0, 0)) >= 0:
    from openai import OpenAI, AsyncOpenAI
    from openai.types import Completion, CompletionChoice, CompletionUsage, CreateEmbeddingResponse, Embedding
    from openai.types.chat import ChatCompletion, ChatCompletionMessage, ChatCompletionChunk
    from openai.types.chat.chat_completion import Choice
    from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice, ChoiceDelta
    from openai.types.chat.chat_completion_message import FunctionCall
    from openai.types.create_embedding_response import Usage

    os.environ['OPENAI_API_KEY'] = 'sk-kkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk'

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

        @patch.object(Uploader, 'upload_span')
        async def test_record(self, mocked_upload_span):
            recorder = OpenAIRecorder()
            recorder.setup()
            proto = signals_pb2.Span()
            context = {}
            recorder.on_span_start(proto, context, DEFAULT_OPTIONS)
            recorder.on_span_stop(proto, context, DEFAULT_OPTIONS)
            recorder.on_span_read(proto, context, DEFAULT_OPTIONS)
            self.assertEqual(proto.libraries[0].name, 'OpenAI Python Library')

        @patch.object(Uploader, 'upload_span')
        async def test_client(self, mocked_upload_span):
            recorder = OpenAIRecorder()
            recorder.setup()
            proto = signals_pb2.Span()

            context = {}
            recorder.on_span_start(proto, context, DEFAULT_OPTIONS)
            recorder.on_span_stop(proto, context, DEFAULT_OPTIONS)
            recorder.on_span_read(proto, context, DEFAULT_OPTIONS)
            self.assertEqual(proto.libraries[0].name, 'OpenAI Python Library')

        @patch.object(Uploader, 'upload_span')
        async def test_completion_create(self, mocked_upload_span):
            recorder = OpenAIRecorder()
            recorder.setup()

            client = OpenAI(
                #api_key="Your_API_Key"
            )

            with patch.object(client.completions, '_post') as mocked_post:
                mocked_post.return_value = Completion(
                    id='cmpl-8Igp0TaBfwDxyxZlB2Lcgrtg89HmQ', 
                    choices=[
                        CompletionChoice(
                            finish_reason='stop', 
                            index=0, 
                            logprobs=None, 
                            text='\n\nWhat is your favorite color?'), 
                        CompletionChoice(
                            finish_reason='stop', 
                            index=1, logprobs=None, 
                            text='\n\nWhat is your favorite color?\n\nMy favorite color is blue.')], 
                    created=1699465202, 
                    model='gpt-3.5-turbo-instruct', 
                    object='text_completion', 
                    system_fingerprint=None, 
                    usage=CompletionUsage(
                        completion_tokens=20, 
                        prompt_tokens=8, 
                        total_tokens=28))

                response = client.completions.create(
                    model="gpt-3.5-turbo-instruct", 
                    prompt=['test prompt 1', 'test prompt 2'],
                    temperature=0.1,
                    top_p=1,
                    max_tokens=1024,
                    frequency_penalty=0,
                    presence_penalty=0)

                recorder.shutdown()

                #print(response.model_dump())

                proto = mocked_upload_span.call_args[0][0]

                self.assertEqual(find_tag(proto, 'component'), 'LLM')
                self.assertEqual(find_tag(proto, 'operation'), 'openai.completions.create')
                self.assertEqual(find_tag(proto, 'endpoint'), 'https://api.openai.com/v1/completions')
                self.assertEqual(find_tag(proto, 'model'), 'gpt-3.5-turbo-instruct')

                self.assertEqual(find_param(proto, 'model'), 'gpt-3.5-turbo-instruct')
                self.assertEqual(find_param(proto, 'max_tokens'), '1024')
                self.assertEqual(find_param(proto, 'temperature'), '0.1')
                self.assertEqual(find_param(proto, 'top_p'), '1')
                self.assertEqual(find_param(proto, 'presence_penalty'), '0')
                self.assertEqual(find_param(proto, 'frequency_penalty'), '0')

                self.assertEqual(find_data_count(proto, 'prompt', 'token_count'), 8)
                self.assertEqual(find_data_count(proto, 'completion', 'token_count'), 20)
                self.assertEqual(find_data_count(proto, 'completion', 'finish_reason_stop'), 2)

                self.assertIsNotNone(find_data_sample(proto, 'prompt'))
                self.assertIsNotNone(find_data_sample(proto, 'completion'))

        @patch.object(Uploader, 'upload_span')
        async def test_completion_create_stream(self, mocked_upload_span):
            recorder = OpenAIRecorder()
            recorder.setup()

            client = OpenAI(
                #api_key="Your_API_Key"
            )

            with patch.object(client.completions, '_post') as mocked_post:
                test_ret = [
                    Completion(
                        id='cmpl-8NH2zI1lzfUs1SId2EDIFAOkcubN8', 
                        choices=[
                            CompletionChoice(
                                finish_reason='stop', 
                                index=0, 
                                logprobs=None, 
                                text=' to')], 
                                created=1700557765, 
                                model='gpt-3.5-turbo-instruct', 
                                object='text_completion', 
                                system_fingerprint=None, 
                                usage=None),
                    Completion(
                        id='cmpl-8NH2zI1lzfUs1SId2EDIFAOkcubN8', 
                        choices=[
                            CompletionChoice(
                                finish_reason='stop', 
                                index=0, 
                                logprobs=None, 
                                text=' ')], 
                                created=1700557765, 
                                model='gpt-3.5-turbo-instruct', 
                                object='text_completion', 
                                system_fingerprint=None, 
                                usage=None)
                ]
                def test_ret_gen():
                    for item in test_ret:
                        yield item
                mocked_post.return_value = test_ret_gen()

                response = client.completions.create(
                    model="gpt-3.5-turbo-instruct", 
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

                proto = mocked_upload_span.call_args[0][0]

                self.assertEqual(find_data_count(proto, 'prompt', 'token_count'), 11)
                self.assertEqual(find_data_count(proto, 'completion', 'finish_reason_stop'), 2)
                self.assertEqual(find_data_count(proto, 'completion', 'token_count'), 2)

                self.assertIsNotNone(find_data_sample(proto, 'prompt'))
                self.assertIsNotNone(find_data_sample(proto, 'completion'))

        @patch.object(Uploader, 'upload_span')
        async def test_completion_create_async(self, mocked_upload_span):
            recorder = OpenAIRecorder()
            recorder.setup()

            client = AsyncOpenAI(
                #api_key="Your_API_Key"
            )

            with patch.object(client.completions, '_post') as mocked_post:
                mocked_post.return_value = Completion(
                    id='cmpl-8Igp0TaBfwDxyxZlB2Lcgrtg89HmQ', 
                    choices=[
                        CompletionChoice(
                            finish_reason='stop', 
                            index=0, 
                            logprobs=None, 
                            text='\n\nWhat is your favorite color?'), 
                        CompletionChoice(
                            finish_reason='stop', 
                            index=1, logprobs=None, 
                            text='\n\nWhat is your favorite color?\n\nMy favorite color is blue.')], 
                    created=1699465202, 
                    model='gpt-3.5-turbo-instruct', 
                    object='text_completion', 
                    system_fingerprint=None, 
                    usage=CompletionUsage(
                        completion_tokens=20, 
                        prompt_tokens=8, 
                        total_tokens=28))

                response = await client.completions.create(
                    model="gpt-3.5-turbo-instruct", 
                    prompt=['test prompt 1', 'test prompt 2'],
                    temperature=0.1,
                    top_p=1,
                    max_tokens=1024,
                    frequency_penalty=0,
                    presence_penalty=0)

                recorder.shutdown()

                #print(response.model_dump())

                proto = mocked_upload_span.call_args[0][0]

                self.assertEqual(find_tag(proto, 'component'), 'LLM')
                self.assertEqual(find_tag(proto, 'operation'), 'openai.completions.create')
                self.assertEqual(find_tag(proto, 'endpoint'), 'https://api.openai.com/v1/completions')
                self.assertEqual(find_tag(proto, 'model'), 'gpt-3.5-turbo-instruct')

                self.assertEqual(find_param(proto, 'model'), 'gpt-3.5-turbo-instruct')
                self.assertEqual(find_param(proto, 'max_tokens'), '1024')
                self.assertEqual(find_param(proto, 'temperature'), '0.1')
                self.assertEqual(find_param(proto, 'top_p'), '1')
                self.assertEqual(find_param(proto, 'presence_penalty'), '0')
                self.assertEqual(find_param(proto, 'frequency_penalty'), '0')

                self.assertEqual(find_data_count(proto, 'prompt', 'token_count'), 8)
                self.assertEqual(find_data_count(proto, 'completion', 'token_count'), 20)
                self.assertEqual(find_data_count(proto, 'completion', 'finish_reason_stop'), 2)

                self.assertIsNotNone(find_data_sample(proto, 'prompt'))
                self.assertIsNotNone(find_data_sample(proto, 'completion'))


        @patch.object(Uploader, 'upload_span')
        async def test_chat_completion_create(self, mocked_upload_span):
            recorder = OpenAIRecorder()
            recorder.setup()

            client = OpenAI(
                #api_key="Your_API_Key"
            )

            with patch.object(client.chat.completions, '_post') as mocked_post:
                mocked_post.return_value = ChatCompletion(
                    id='chatcmpl-8NeQhqaA8Cq3fVeJpznStwgC4hWxt', 
                    choices=[
                        Choice(
                            finish_reason='stop', 
                            index=0, 
                            message=ChatCompletionMessage(
                                content='What are the benefits of regular exercise?', 
                                role='assistant', 
                                function_call=None, 
                                tool_calls=None))], 
                                created=1700647647, 
                                model='gpt-3.5-turbo-0613', 
                                object='chat.completion', 
                                system_fingerprint=None, 
                                usage=CompletionUsage(
                                    completion_tokens=8, 
                                    prompt_tokens=19, 
                                    total_tokens=27))

                response = client.chat.completions.create(
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

                proto = mocked_upload_span.call_args[0][0]

                self.assertEqual(find_tag(proto, 'component'), 'LLM')
                self.assertEqual(find_tag(proto, 'operation'), 'openai.chat.completions.create')
                self.assertEqual(find_tag(proto, 'endpoint'), 'https://api.openai.com/v1/chat/completions')
                self.assertEqual(find_tag(proto, 'model'), 'gpt-3.5-turbo')

                self.assertEqual(find_param(proto, 'model'), 'gpt-3.5-turbo')
                self.assertEqual(find_param(proto, 'max_tokens'), '1024')
                self.assertEqual(find_param(proto, 'temperature'), '0.1')
                self.assertEqual(find_param(proto, 'top_p'), '1')
                self.assertEqual(find_param(proto, 'presence_penalty'), '0')
                self.assertEqual(find_param(proto, 'frequency_penalty'), '0')

                self.assertEqual(find_data_count(proto, 'messages', 'token_count'), 19)
                self.assertEqual(find_data_count(proto, 'completion', 'token_count'), 8)
                self.assertEqual(find_data_count(proto, 'completion', 'finish_reason_stop'), 1.0)

                self.assertIsNotNone(find_data_sample(proto, 'messages'))
                self.assertIsNotNone(find_data_sample(proto, 'completion'))


        @patch.object(Uploader, 'upload_span')
        async def test_chat_completion_create_function(self, mocked_upload_span):
            recorder = OpenAIRecorder()
            recorder.setup()

            client = OpenAI(
                #api_key="Your_API_Key"
            )

            with patch.object(client.chat.completions, '_post') as mocked_post:
                mocked_post.return_value = ChatCompletion(
                    id='chatcmpl-8NjG5hzrQXtbnnYFsR3t3siBRcFWl', 
                    choices=[
                        Choice(
                            finish_reason='function_call', 
                            index=0, 
                            message=ChatCompletionMessage(
                                content=None, 
                                role='assistant', 
                                function_call=FunctionCall(
                                    arguments='{\n  "location": "Boston, MA"\n}', 
                                    name='get_current_weather'), tool_calls=None))], 
                                    created=1700666209, 
                                    model='gpt-3.5-turbo-0613', 
                                    object='chat.completion', 
                                    system_fingerprint=None, 
                                    usage=CompletionUsage(
                                        completion_tokens=18, 
                                        prompt_tokens=78, 
                                        total_tokens=96))

                response = client.chat.completions.create(
                    model="gpt-3.5-turbo-0613", 
                    messages=[{'role': 'system', 'content': 'What\'s the weather like in Boston?'}],
                    functions=[
                        {
                            "name": "get_current_weather",
                            "description": "Get the current weather in a given location",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "location": {
                                        "type": "string",
                                        "description": "The city and state, e.g. San Francisco, CA",
                                    },
                                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                                },
                                "required": ["location"],
                            },
                        }
                    ],
                    function_call="auto",
                    temperature=0.1,
                    top_p=1,
                    max_tokens=1024,
                    frequency_penalty=0,
                    presence_penalty=0)

                recorder.shutdown()

                #print(response)

                proto = mocked_upload_span.call_args[0][0]

                self.assertEqual(find_tag(proto, 'component'), 'LLM')
                self.assertEqual(find_tag(proto, 'operation'), 'openai.chat.completions.create')
                self.assertEqual(find_tag(proto, 'endpoint'), 'https://api.openai.com/v1/chat/completions')
                self.assertEqual(find_tag(proto, 'model'), 'gpt-3.5-turbo-0613')

                self.assertEqual(find_param(proto, 'model'), 'gpt-3.5-turbo-0613')
                self.assertEqual(find_param(proto, 'function_call'), 'auto')
                self.assertEqual(find_param(proto, 'max_tokens'), '1024')
                self.assertEqual(find_param(proto, 'temperature'), '0.1')
                self.assertEqual(find_param(proto, 'top_p'), '1')
                self.assertEqual(find_param(proto, 'presence_penalty'), '0')
                self.assertEqual(find_param(proto, 'frequency_penalty'), '0')

                self.assertEqual(find_data_count(proto, 'messages', 'token_count'), 78)
                self.assertEqual(find_data_count(proto, 'completion', 'token_count'), 18)
                self.assertEqual(find_data_count(proto, 'completion', 'finish_reason_function_call'), 1.0)

                self.assertIsNotNone(find_data_sample(proto, 'messages'))
                self.assertIsNotNone(find_data_sample(proto, 'functions'))
                self.assertIsNotNone(find_data_sample(proto, 'completion'))


        @patch.object(Uploader, 'upload_span')
        async def test_chat_completion_create_stream(self, mocked_upload_span):
            recorder = OpenAIRecorder()
            recorder.setup()

            client = OpenAI(
                #api_key="Your_API_Key"
            )

            with patch.object(client.chat.completions, '_post') as mocked_post:
                test_ret = [
                    ChatCompletionChunk(
                        id='chatcmpl-8O3j3TnSiuvjZTBeBFx3aWA9Ukh2x',
                        choices=[
                            ChunkChoice(
                                delta=ChoiceDelta(
                                    content='', 
                                    function_call=None, 
                                    role='assistant', 
                                    tool_calls=None), 
                                finish_reason='stop', 
                                index=0)], 
                            created=1700744885, 
                            model='gpt-4-0613', 
                            object='chat.completion.chunk'),
                    ChatCompletionChunk(
                        id='chatcmpl-8O3j3TnSiuvjZTBeBFx3aWA9Ukh2x', 
                        choices=[
                            ChunkChoice(
                                delta=ChoiceDelta(
                                    content='We', 
                                    function_call=None, 
                                    role=None, 
                                    tool_calls=None), 
                                finish_reason='stop', 
                                index=0)], 
                            created=1700744885, 
                            model='gpt-4-0613', 
                            object='chat.completion.chunk')
                ]
                def test_ret_gen():
                    for item in test_ret:
                        yield item
                mocked_post.return_value = test_ret_gen()

                response = client.chat.completions.create(
                    model="gpt-4", 
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful.",
                        },
                        {
                            "role": "system",
                            "name": "example_user",
                            "content": "New synergies will help drive top-line growth.",
                        },
                        {
                            "role": "user",
                            "name": "example",
                            "content": "This late pivot means.",
                        }
                    ],
                    temperature=0.1,
                    top_p=1,
                    max_tokens=1024,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stream=True)

                for r in response:
                    pass

                recorder.shutdown()

                proto = mocked_upload_span.call_args[0][0]

                self.assertEqual(find_data_count(proto, 'messages', 'token_count'), 40)
                self.assertEqual(find_data_count(proto, 'completion', 'finish_reason_stop'), 2)
                self.assertEqual(find_data_count(proto, 'completion', 'token_count'), 2)

                self.assertIsNotNone(find_data_sample(proto, 'messages'))
                self.assertIsNotNone(find_data_sample(proto, 'completion'))


        @patch.object(Uploader, 'upload_span')
        async def test_chat_completion_create_stream_async(self, mocked_upload_span):
            recorder = OpenAIRecorder()
            recorder.setup()

            client = AsyncOpenAI(
                #api_key="Your_API_Key"
            )

            with patch.object(client.chat.completions, '_post') as mocked_post:
                test_ret = [
                    ChatCompletionChunk(
                        id='chatcmpl-8OQctcUZdbOT1KO8x2IKGUQE7G33b', 
                        choices=[ChunkChoice(
                            delta=ChoiceDelta(
                                content='', 
                                function_call=None, 
                                role='assistant', 
                                tool_calls=None), 
                            finish_reason='stop', 
                            index=0)], 
                        created=1700832915, 
                        model='gpt-4-0613', 
                        object='chat.completion.chunk'),
                    ChatCompletionChunk(
                        id='chatcmpl-8OQctcUZdbOT1KO8x2IKGUQE7G33b', 
                        choices=[ChunkChoice(
                            delta=ChoiceDelta(
                                content='We', 
                                function_call=None, 
                                role=None, 
                                tool_calls=None), 
                            finish_reason='stop', index=0)], 
                        created=1700832915, 
                        model='gpt-4-0613', 
                        object='chat.completion.chunk')                    
                ]

                async def test_ret_gen():
                    for item in test_ret:
                        yield item
                mocked_post.return_value = test_ret_gen()

                response = await client.chat.completions.create(
                    model="gpt-4", 
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful.",
                        },
                        {
                            "role": "system",
                            "name": "example_user",
                            "content": "New synergies will help drive top-line growth.",
                        },
                        {
                            "role": "user",
                            "name": "example",
                            "content": "This late pivot means.",
                        }
                    ],
                    temperature=0.1,
                    top_p=1,
                    max_tokens=1024,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stream=True)

                async for r in response:
                    pass

                recorder.shutdown()

                proto = mocked_upload_span.call_args[0][0]

                self.assertEqual(find_data_count(proto, 'messages', 'token_count'), 40)
                self.assertEqual(find_data_count(proto, 'completion', 'finish_reason_stop'), 2)
                self.assertEqual(find_data_count(proto, 'completion', 'token_count'), 2)

                self.assertIsNotNone(find_data_sample(proto, 'messages'))
                self.assertIsNotNone(find_data_sample(proto, 'completion'))

        @patch.object(Uploader, 'upload_span')
        async def test_embedding_create(self, mocked_upload_span):
            recorder = OpenAIRecorder()
            recorder.setup()

            client = OpenAI(
                #api_key="Your_API_Key"
            )

            with patch.object(client.embeddings, '_post') as mocked_post:
                mocked_post.return_value = CreateEmbeddingResponse(
                    data=[Embedding(
                        embedding=[0.0008772446890361607, -0.0045392164029181], 
                        index=1, 
                        object='embedding')], 
                    model='text-embedding-ada-002-v2', 
                    object='list', 
                    usage=Usage(
                        prompt_tokens=8, 
                        total_tokens=8))

                response = client.embeddings.create(
                    model="text-embedding-ada-002",
                    input=['test text 1', 'test text 2'])

                recorder.shutdown()

                proto = mocked_upload_span.call_args[0][0]

                self.assertEqual(find_tag(proto, 'component'), 'LLM')
                self.assertEqual(find_tag(proto, 'operation'), 'openai.embeddings.create')
                self.assertEqual(find_tag(proto, 'endpoint'), 'https://api.openai.com/v1/embeddings')
                self.assertEqual(find_tag(proto, 'model'), 'text-embedding-ada-002-v2')

                self.assertEqual(find_param(proto, 'model'), 'text-embedding-ada-002')

                self.assertEqual(find_data_count(proto, 'input', 'token_count'), 8.0)

                self.assertIsNotNone(find_data_sample(proto, 'input'))
                self.assertIsNotNone(find_data_sample(proto, 'embeddings'))
