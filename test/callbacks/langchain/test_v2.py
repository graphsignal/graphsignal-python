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
from langchain.agents import Tool, initialize_agent, load_tools
from typing import Any, List, Mapping, Optional
from langchain.llms.base import LLM
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

import graphsignal
from graphsignal.uploader import Uploader
from graphsignal.callbacks.langchain.v2 import GraphsignalCallbackHandler
from graphsignal.recorders.openai_recorder import OpenAIRecorder
from graphsignal.proto_utils import find_tag, find_data_count, find_data_sample

logger = logging.getLogger('graphsignal')


class DummyLLM(LLM):
    @property
    def _llm_type(self) -> str:
        return "dummy"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        return 'Final Answer:42'

    async def _acall(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        return 'Final Answer:42'

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {}


class GraphsignalCallbackHandlerTest(unittest.IsolatedAsyncioTestCase):
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
    async def test_chain(self, mocked_upload_span):
        graphsignal.set_context_tag('ct1', 'v1')

        llm = DummyLLM()
        tools = load_tools(["llm-math"], llm=llm)
        agent = initialize_agent(
            tools, llm, agent="zero-shot-react-description", verbose=True)
        agent.run("What is 2 raised to .123243 power?")

        t3 = mocked_upload_span.call_args_list[0][0][0]
        t2 = mocked_upload_span.call_args_list[1][0][0]
        t1 = mocked_upload_span.call_args_list[2][0][0]

        self.assertEqual(t1.labels, ['root'])
        self.assertEqual(find_tag(t1, 'ct1'), 'v1')
        self.assertEqual(find_tag(t1, 'component'), 'Agent')
        self.assertEqual(find_tag(t1, 'operation'), 'langchain.agents.agent.AgentExecutor')
        self.assertIsNotNone(find_data_sample(t1, 'inputs'))
        self.assertIsNotNone(find_data_sample(t1, 'outputs'))

        self.assertEqual(t2.labels, [])
        self.assertEqual(find_tag(t2, 'ct1'), 'v1')
        self.assertEqual(find_tag(t2, 'operation'), 'langchain.chains.llm.LLMChain')
        self.assertEqual(t2.context.parent_span_id, t1.span_id)
        self.assertEqual(t2.context.root_span_id, t1.span_id)
        self.assertIsNotNone(find_data_sample(t2, 'inputs'))
        self.assertIsNotNone(find_data_sample(t2, 'outputs'))

        self.assertEqual(t3.labels, [])
        self.assertEqual(find_tag(t3, 'ct1'), 'v1')
        self.assertEqual(find_tag(t3, 'component'), 'LLM')
        self.assertEqual(find_tag(t3, 'operation'), 'test.callbacks.langchain.test_v2.DummyLLM')
        self.assertEqual(t3.context.parent_span_id, t2.span_id)
        self.assertEqual(t3.context.root_span_id, t1.span_id)

    @patch.object(Uploader, 'upload_span')
    async def test_chain_async(self, mocked_upload_span):
        graphsignal.set_context_tag('ct1', 'v1')

        llm = DummyLLM()
        tools = load_tools(["llm-math"], llm=llm)
        agent = initialize_agent(
            tools, llm, agent="zero-shot-react-description", verbose=True)
        await agent.arun("What is 2 raised to .123243 power?")

        t3 = mocked_upload_span.call_args_list[0][0][0]
        t2 = mocked_upload_span.call_args_list[1][0][0]
        t1 = mocked_upload_span.call_args_list[2][0][0]

        self.assertEqual(t1.labels, ['root'])
        self.assertEqual(find_tag(t1, 'ct1'), 'v1')
        self.assertEqual(find_tag(t1, 'component'), 'Agent')
        self.assertEqual(find_tag(t1, 'operation'), 'langchain.agents.agent.AgentExecutor')
        self.assertIsNotNone(find_data_sample(t1, 'inputs'))
        self.assertIsNotNone(find_data_sample(t1, 'outputs'))

        self.assertEqual(t2.labels, [])
        self.assertEqual(find_tag(t2, 'ct1'), 'v1')
        self.assertEqual(find_tag(t2, 'component'), 'Agent')
        self.assertEqual(find_tag(t2, 'operation'), 'langchain.chains.llm.LLMChain')
        self.assertEqual(t2.context.parent_span_id, t1.span_id)
        self.assertEqual(t2.context.root_span_id, t1.span_id)
        self.assertIsNotNone(find_data_sample(t2, 'inputs'))
        self.assertIsNotNone(find_data_sample(t2, 'outputs'))

        self.assertEqual(t3.labels, [])
        self.assertEqual(find_tag(t3, 'ct1'), 'v1')
        self.assertEqual(find_tag(t3, 'component'), 'LLM')
        self.assertEqual(find_tag(t3, 'operation'), 'test.callbacks.langchain.test_v2.DummyLLM')
        self.assertEqual(t3.context.parent_span_id, t2.span_id)
        self.assertEqual(t3.context.root_span_id, t1.span_id)


    @patch.object(Uploader, 'upload_span')
    async def test_chain_async_with_decorator(self, mocked_upload_span):
        prompt = PromptTemplate(
            input_variables=["product"],
            template="What is a good name for a company that makes {product}?",
        )

        llm = DummyLLM()
        chain = LLMChain(llm=llm, prompt=prompt)

        @graphsignal.trace_function
        async def run_chain():
            await chain.arun("colorful socks")

        await run_chain()

        t3 = mocked_upload_span.call_args_list[0][0][0]
        t2 = mocked_upload_span.call_args_list[1][0][0]
        t1 = mocked_upload_span.call_args_list[2][0][0]

        self.assertEqual(t1.labels, ['root'])
        self.assertEqual(find_tag(t1, 'operation'), 'run_chain')

        self.assertEqual(t2.labels, [])
        self.assertEqual(find_tag(t2, 'operation'), 'langchain.chains.llm.LLMChain')
        self.assertEqual(t2.context.parent_span_id, t1.span_id)
        self.assertEqual(t2.context.root_span_id, t1.span_id)

        self.assertEqual(t3.labels, [])
        self.assertEqual(find_tag(t3, 'component'), 'LLM')
        self.assertEqual(find_tag(t3, 'operation'), 'test.callbacks.langchain.test_v2.DummyLLM')
        self.assertEqual(t3.context.parent_span_id, t2.span_id)
        self.assertEqual(t3.context.root_span_id, t1.span_id)

    @patch.object(Uploader, 'upload_span')
    @patch.object(openai.ChatCompletion, 'acreate')
    async def test_llm_async(self, mocked_acreate, mocked_upload_span):
        os.environ['OPENAI_API_KEY'] = 'fake-key'

        # mocking overrides autoinstrumentation, reinstrument
        recorder = OpenAIRecorder()
        recorder.setup()

        test_ret = [
            {
                "choices": [
                    {
                        "finish_reason": None,
                        "index": 0,
                        "logprobs": None,
                        "delta": {
                            "role": "assistant",
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
                        "index": 0,
                        "logprobs": None,
                        "delta": {
                            "role": "assistant",
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
        async def test_ret_gen():
            for item in test_ret:
                yield item
        mocked_acreate.return_value = test_ret_gen()

        graphsignal.set_context_tag('ct1', 'v1')

        llm = ChatOpenAI(
            verbose=True,
            temperature=0,
            streaming=True)

        async with graphsignal.start_trace('test'):
            await llm.agenerate([[HumanMessage(content='What is 2 raised to .123243 power?')]])

        t3 = mocked_upload_span.call_args_list[0][0][0]
        t2 = mocked_upload_span.call_args_list[1][0][0]
        t1 = mocked_upload_span.call_args_list[2][0][0]

        self.assertEqual(t1.labels, ['root'])
        self.assertEqual(find_tag(t1, 'ct1'), 'v1')
        self.assertEqual(find_tag(t1, 'operation'), 'test')

        self.assertEqual(t2.labels, [])
        self.assertEqual(find_tag(t2, 'ct1'), 'v1')
        self.assertEqual(find_tag(t2, 'component'), 'LLM')
        self.assertEqual(find_tag(t2, 'operation'), 'langchain.chat_models.openai.ChatOpenAI')
        self.assertIsNotNone(find_data_sample(t2, 'messages'))

        self.assertEqual(t3.labels, [])
        self.assertEqual(find_tag(t3, 'ct1'), 'v1')
        self.assertEqual(find_tag(t3, 'component'), 'LLM')
        self.assertEqual(find_tag(t3, 'operation'), 'openai.ChatCompletion.acreate')
        self.assertEqual(find_data_count(t3, 'messages', 'token_count'), 19.0)
        self.assertEqual(find_data_count(t3, 'completion', 'token_count'), 2.0)
