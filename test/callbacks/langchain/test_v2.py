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
from langchain.agents import AgentExecutor, create_react_agent, load_tools
from typing import Any, List, Mapping, Optional
from langchain.llms.base import LLM
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain import hub

import graphsignal
from graphsignal.uploader import Uploader
from graphsignal.callbacks.langchain.v2 import GraphsignalCallbackHandler
from graphsignal.recorders.openai_recorder import OpenAIRecorder
from test.proto_utils import find_tag, find_usage, find_payload

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
        prompt = hub.pull("hwchase17/react")
        agent = create_react_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
        agent_executor.invoke({"input": "What is 2 raised to .123243 power?"})

        #print(mocked_upload_span.call_args_list)

        t1 = mocked_upload_span.call_args_list[0][0][0]

        self.assertEqual(find_tag(t1, 'ct1'), 'v1')
        self.assertEqual(find_tag(t1, 'library'), 'langchain')
        self.assertEqual(find_tag(t1, 'operation'), 'test.callbacks.langchain.test_v2.DummyLLM')

    @patch.object(Uploader, 'upload_span')
    async def test_chain_async(self, mocked_upload_span):
        graphsignal.set_context_tag('ct1', 'v1')

        llm = DummyLLM()
        tools = load_tools(["llm-math"], llm=llm)
        prompt = hub.pull("hwchase17/react")
        agent = create_react_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
        await agent_executor.ainvoke({"input": "What is 2 raised to .123243 power?"})

        #print(mocked_upload_span.call_args_list)

        t1 = mocked_upload_span.call_args_list[0][0][0]

        self.assertEqual(find_tag(t1, 'ct1'), 'v1')
        self.assertEqual(find_tag(t1, 'library'), 'langchain')
        self.assertEqual(find_tag(t1, 'operation'), 'test.callbacks.langchain.test_v2.DummyLLM')


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
            await chain.ainvoke("colorful socks")

        await run_chain()

        t2 = mocked_upload_span.call_args_list[0][0][0]
        t1 = mocked_upload_span.call_args_list[1][0][0]
        t0 = mocked_upload_span.call_args_list[2][0][0]

        self.assertEqual(find_tag(t0, 'operation'), 'run_chain')

        self.assertEqual(find_tag(t1, 'operation'), 'langchain.chains.llm.LLMChain')
        self.assertEqual(t1.context.parent_span_id, t0.span_id)
        self.assertEqual(t1.context.root_span_id, t0.span_id)

        self.assertEqual(find_tag(t2, 'library'), 'langchain')
        self.assertEqual(find_tag(t2, 'operation'), 'test.callbacks.langchain.test_v2.DummyLLM')
        self.assertEqual(t2.context.parent_span_id, t1.span_id)
        self.assertEqual(t2.context.root_span_id, t0.span_id)
