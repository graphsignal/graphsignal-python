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
import openai

import graphsignal
from graphsignal import client
from graphsignal.uploader import Uploader
from graphsignal.recorders.openai_recorder import OpenAIRecorder
from test.model_utils import find_tag, find_usage, find_payload

logger = logging.getLogger('graphsignal')


from openai import OpenAI, AsyncOpenAI
from openai.types import Completion, CompletionChoice, CompletionUsage, CreateEmbeddingResponse, Embedding
from openai.types.chat import ChatCompletion, ChatCompletionMessage, ChatCompletionChunk
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice, ChoiceDelta
from openai.types.chat.chat_completion_message import FunctionCall
from openai.types.create_embedding_response import Usage

os.environ['OPENAI_API_KEY'] = 'sk-kkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk'
os.environ['LANGCHAIN_API_KEY'] = 'kkk'

class OpenAIRecorderTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.maxDiff = None
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
        model = client.Span(
            span_id='s1',
            start_us=0,
            end_us=0,
            config=[]
        )
        context = {}
        recorder.on_span_start(model, context)
        recorder.on_span_stop(model, context)
        recorder.on_span_read(model, context)
        self.assertEqual(model.config[0].key, 'openai.library.version')

    @patch.object(Uploader, 'upload_span')
    async def test_client(self, mocked_upload_span):
        recorder = OpenAIRecorder()
        recorder.setup()
        model = client.Span(
            span_id='s1',
            start_us=0,
            end_us=0,
            config=[]
        )
        context = {}
        recorder.on_span_start(model, context)
        recorder.on_span_stop(model, context)
        recorder.on_span_read(model, context)
        self.assertEqual(model.config[0].key, 'openai.library.version')

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
                presence_penalty=0,
                user='u1',
                extra_headers={
                    'Graphsignal-Tags': ' k1=v1, k2=v2'
                })

            recorder.shutdown()

            model = mocked_upload_span.call_args[0][0]

            self.assertEqual(find_tag(model, 'model_type'), 'chat')
            self.assertEqual(find_tag(model, 'library'), 'openai')
            self.assertEqual(find_tag(model, 'operation'), 'openai.chat.completions.create')
            self.assertEqual(find_tag(model, 'endpoint'), 'https://api.openai.com/v1/chat/completions')
            self.assertEqual(find_tag(model, 'model'), 'gpt-3.5-turbo')
            self.assertEqual(find_tag(model, 'k1'), 'v1')
            self.assertEqual(find_tag(model, 'user_id'), 'u1')

            self.assertEqual(find_usage(model, 'input', 'token_count'), 19)
            self.assertEqual(find_usage(model, 'output', 'token_count'), 8)

            self.assertIsNotNone(find_payload(model, 'input'))
            self.assertIsNotNone(find_payload(model, 'output'))


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

            model = mocked_upload_span.call_args[0][0]

            self.assertEqual(find_tag(model, 'model_type'), 'chat')
            self.assertEqual(find_tag(model, 'operation'), 'openai.chat.completions.create')
            self.assertEqual(find_tag(model, 'api_provider'), 'openai')
            self.assertEqual(find_tag(model, 'endpoint'), 'https://api.openai.com/v1/chat/completions')
            self.assertEqual(find_tag(model, 'model'), 'gpt-3.5-turbo-0613')

            self.assertEqual(find_usage(model, 'input', 'token_count'), 78)
            self.assertEqual(find_usage(model, 'output', 'token_count'), 18)

            self.assertIsNotNone(find_payload(model, 'input'))
            self.assertIsNotNone(find_payload(model, 'output'))


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

            model = mocked_upload_span.call_args[0][0]

            self.assertEqual(find_usage(model, 'input', 'token_count'), 40)
            self.assertEqual(find_usage(model, 'output', 'token_count'), 2)

            self.assertIsNotNone(find_payload(model, 'input'))
            self.assertIsNotNone(find_payload(model, 'output'))


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

            model = mocked_upload_span.call_args[0][0]


            self.assertEqual(find_tag(model, 'model_type'), 'chat')
            self.assertEqual(find_tag(model, 'model'), 'gpt-4')
            self.assertEqual(find_tag(model, 'endpoint'), 'https://api.openai.com/v1/chat/completions')

            self.assertEqual(find_usage(model, 'input', 'token_count'), 40)
            self.assertEqual(find_usage(model, 'output', 'token_count'), 2)

            input_json = json.loads(base64.b64decode(find_payload(model, 'input').content_base64))
            #pp = pprint.PrettyPrinter()
            #pp.pprint(input_json)

            self.assertEqual(input_json, 
                {'frequency_penalty': 0,
                'max_tokens': 1024,
                'messages': [{'content': 'You are a helpful.', 'role': 'system'},
                            {'content': 'New synergies will help drive top-line growth.',
                            'name': 'example_user',
                            'role': 'system'},
                            {'content': 'This late pivot means.',
                            'name': 'example',
                            'role': 'user'}],
                'model': 'gpt-4',
                'presence_penalty': 0,
                'stream': True,
                'temperature': 0.1,
                'top_p': 1})

            output_json = json.loads(base64.b64decode(find_payload(model, 'output').content_base64))
            #pp = pprint.PrettyPrinter()
            #pp.pprint(output_json)

            self.assertEqual(output_json, [{
                'choices': [{'delta': {'content': '',
                                        'function_call': None,
                                        'refusal': None,
                                        'role': 'assistant',
                                        'tool_calls': None},
                            'finish_reason': 'stop',
                            'index': 0,
                            'logprobs': None}],
                'created': 1700832915,
                'id': 'chatcmpl-8OQctcUZdbOT1KO8x2IKGUQE7G33b',
                'model': 'gpt-4-0613',
                'object': 'chat.completion.chunk',
                'service_tier': None,
                'system_fingerprint': None,
                'usage': None},
                {'choices': [{'delta': {'content': 'We',
                                        'function_call': None,
                                        'refusal': None,
                                        'role': None,
                                        'tool_calls': None},
                            'finish_reason': 'stop',
                            'index': 0,
                            'logprobs': None}],
                'created': 1700832915,
                'id': 'chatcmpl-8OQctcUZdbOT1KO8x2IKGUQE7G33b',
                'model': 'gpt-4-0613',
                'object': 'chat.completion.chunk',
                'service_tier': None,
                'system_fingerprint': None,
                'usage': None}])

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

            model = mocked_upload_span.call_args[0][0]

            self.assertEqual(find_tag(model, 'operation'), 'openai.embeddings.create')
            self.assertEqual(find_tag(model, 'endpoint'), 'https://api.openai.com/v1/embeddings')
            self.assertEqual(find_tag(model, 'model'), 'text-embedding-ada-002-v2')

            self.assertEqual(find_usage(model, 'input', 'token_count'), 8.0)

            input_json = json.loads(base64.b64decode(find_payload(model, 'input').content_base64))

            #pp = pprint.PrettyPrinter()
            #pp.pprint(input_json)

            self.assertEqual(input_json, {'input': ['test text 1', 'test text 2'], 'model': 'text-embedding-ada-002'})
