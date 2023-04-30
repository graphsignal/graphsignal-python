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
import langchain
from langchain.agents import Tool, initialize_agent, load_tools
from langchain.llms import OpenAI
from typing import Any, List, Mapping, Optional
from langchain.llms.base import LLM
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.callbacks.base import AsyncCallbackHandler, AsyncCallbackManager

import graphsignal
from graphsignal.proto import signals_pb2
from graphsignal.uploader import Uploader
from graphsignal.traces import DEFAULT_OPTIONS
from graphsignal.recorders.langchain_recorder import LangChainRecorder
from graphsignal.recorders.openai_recorder import OpenAIRecorder
from graphsignal.callbacks.langchain import GraphsignalAsyncCallbackHandler

logger = logging.getLogger('graphsignal')


class DummyLLM(LLM):
    @property
    def _llm_type(self) -> str:
        return "dummy"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        return 'Final Answer:42'

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {}


class ExtraCallbackHandler(AsyncCallbackHandler):
    def __init__(self):
        pass

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        pass


class LangChainRecorderTest(unittest.IsolatedAsyncioTestCase):
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
        recorder = LangChainRecorder()
        recorder.setup()
        proto = signals_pb2.Trace()
        context = {}
        recorder.on_trace_start(proto, context, DEFAULT_OPTIONS)
        recorder.on_trace_stop(proto, context, DEFAULT_OPTIONS)
        recorder.on_trace_read(proto, context, DEFAULT_OPTIONS)

        self.assertEqual(proto.frameworks[0].name, 'LangChain')

    @patch.object(Uploader, 'upload_trace')
    async def test_chain(self, mocked_upload_trace):
        llm = OpenAI(temperature=0)
        llm = DummyLLM()
        tools = load_tools(["llm-math"], llm=llm)
        agent = initialize_agent(
            tools, llm, agent="zero-shot-react-description", verbose=True
        )
        agent.run("What is 2 raised to .123243 power?")

        t3 = mocked_upload_trace.call_args_list[0][0][0]
        t2 = mocked_upload_trace.call_args_list[1][0][0]
        t1 = mocked_upload_trace.call_args_list[2][0][0]

        #pp = pprint.PrettyPrinter()
        #pp.pprint(MessageToJson(proto))

        self.assertEqual(t1.frameworks[0].name, 'OpenAI Python Library')
        self.assertEqual(t1.frameworks[1].name, 'LangChain')
        self.assertEqual(t1.labels, ['root'])

        self.assertEqual(find_data_count(t1, 'inputs', 'byte_count'), 34.0)
        self.assertEqual(find_data_count(t1, 'inputs', 'element_count'), 1.0)
        self.assertEqual(find_data_count(t1, 'outputs', 'byte_count'), 2.0)
        self.assertEqual(find_data_count(t1, 'outputs', 'element_count'), 1.0)

        self.assertEqual(find_tag(t1, 'component'), 'Agent')
        self.assertEqual(find_tag(t1, 'operation'), 'langchain.chains.AgentExecutor')

        self.assertEqual(find_tag(t2, 'operation'), 'langchain.chains.LLMChain')
        self.assertEqual(t2.span.parent_trace_id, t1.trace_id)
        self.assertEqual(t2.span.root_trace_id, t1.trace_id)

        self.assertEqual(find_tag(t3, 'component'), 'LLM')
        self.assertEqual(find_tag(t3, 'operation'), 'langchain.llms.DummyLLM')
        self.assertEqual(t3.span.parent_trace_id, t2.trace_id)
        self.assertEqual(t3.span.root_trace_id, t1.trace_id)

    @patch.object(Uploader, 'upload_trace')
    @patch.object(openai.ChatCompletion, 'acreate')
    async def test_chain_async(self, mocked_acreate, mocked_upload_trace):
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

        llm = ChatOpenAI(
            verbose=True,
            temperature=0,
            streaming=True,
            callback_manager=AsyncCallbackManager([ExtraCallbackHandler(), GraphsignalAsyncCallbackHandler()]),
        )

        with graphsignal.start_trace('test'):
            await llm.agenerate([[HumanMessage(content='What is 2 raised to .123243 power?')]])

        t2 = mocked_upload_trace.call_args_list[0][0][0]
        t1 = mocked_upload_trace.call_args_list[1][0][0]

        #pp = pprint.PrettyPrinter()
        #pp.pprint(MessageToJson(t1))

        self.assertEqual(t1.labels, ['root'])
        self.assertEqual(find_tag(t1, 'operation'), 'test')

        self.assertEqual(t2.frameworks[0].name, 'OpenAI Python Library')
        self.assertEqual(t2.frameworks[1].name, 'LangChain')
        self.assertEqual(find_tag(t2, 'component'), 'LLM')
        self.assertEqual(find_tag(t2, 'operation'), 'openai.ChatCompletion.acreate')
        self.assertEqual(find_data_count(t2, 'messages', 'byte_count'), 38.0)
        self.assertEqual(find_data_count(t2, 'messages', 'element_count'), 2.0)
        self.assertEqual(find_data_count(t2, 'messages', 'token_count'), 19.0)
        self.assertEqual(find_data_count(t2, 'completion', 'byte_count'), 4.0)
        self.assertEqual(find_data_count(t2, 'completion', 'element_count'), 2.0)
        self.assertEqual(find_data_count(t2, 'completion', 'token_count'), 2.0)


def find_tag(proto, key):
    for tag in proto.tags:
        if tag.key == key:
            return tag.value
    return None


def find_data_count(proto, data_name, count_name):
    for data_stats in proto.data_profile:
        if data_stats.data_name == data_name:
            for data_count in data_stats.counts:
                if data_count.name == count_name:
                    return data_count.count
    return None