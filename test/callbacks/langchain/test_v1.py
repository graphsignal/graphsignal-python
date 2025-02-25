import unittest
import logging
import sys
import os
import json
import time
from unittest.mock import patch, Mock
import pprint
import openai
from langchain_core.tools import tool
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import AgentExecutor, create_react_agent
from typing import Any, List, Mapping, Optional
from langchain.llms.base import BaseLLM
from langchain_community.llms.fake import FakeListLLM
from langchain.schema import LLMResult, Generation
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain import hub
from langchain_core.messages import HumanMessage

import graphsignal
from graphsignal.uploader import Uploader
from graphsignal.recorders.openai_recorder import OpenAIRecorder
from test.model_utils import find_tag, find_usage, find_payload

logger = logging.getLogger('graphsignal')

@tool
def multiply(first_int: int, second_int: int) -> int:
    """Multiply two integers together."""
    return first_int * second_int

class GraphsignalCallbackHandlerTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        if len(logger.handlers) == 0:
            logger.addHandler(logging.StreamHandler(sys.stdout))
        graphsignal.configure(
            api_key='k1',
            debug_mode=True)
        graphsignal._tracer.export_on_shutdown = False

    async def asyncTearDown(self):
        graphsignal.shutdown()


    @patch.object(Uploader, 'upload_span')
    async def test_callback_tags(self, mocked_upload_span):
        from graphsignal.callbacks.langchain import GraphsignalCallbackHandler
        llm = FakeListLLM(
            responses=['Final Answer:42'],
            callbacks=[GraphsignalCallbackHandler(tags=dict(k1='v1'))]
        )

        llm.invoke([HumanMessage(content="Tell me a joke")])

        t1 = mocked_upload_span.call_args_list[0][0][0]

        self.assertEqual(find_tag(t1, 'k1'), 'v1')


    @patch.object(Uploader, 'upload_span')
    @patch('graphsignal.callbacks.langchain.v1.uuid_sha1', return_value='s1')
    async def test_chain(self, mocked_uuid_sha1, mocked_upload_span):
        os.environ["LANGCHAIN_TRACING_V2"] = "false"

        graphsignal.set_context_tag('ct1', 'v1')

        llm = FakeListLLM(responses=['Final Answer:42'])
        
        tools = [multiply]
        prompt = hub.pull("hwchase17/react")
        agent = create_react_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
        agent_executor.invoke({"input": "What is 2 times 4?"})

        def find_span(op_name):
            for call in mocked_upload_span.call_args_list:
                check_op_name = find_tag(call[0][0], 'operation')
                if check_op_name == op_name:
                    return call[0][0]

        llm_span = find_span('langchain_community.llms.fake.FakeListLLM')
        self.assertEqual(find_tag(llm_span, 'model_type'), 'chat')
        self.assertEqual(find_tag(llm_span, 'ct1'), 'v1')

    @patch.object(Uploader, 'upload_span')
    @patch('graphsignal.callbacks.langchain.v1.uuid_sha1', return_value='s1')
    async def test_chain_async(self, mocked_uuid_sha1, mocked_upload_span):
        os.environ["LANGCHAIN_TRACING_V2"] = "false"

        graphsignal.set_context_tag('ct1', 'v1')

        llm = FakeListLLM(responses=['Final Answer:42'])
        tools = [multiply]
        prompt = hub.pull("hwchase17/react")
        agent = create_react_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
        await agent_executor.ainvoke({"input": "What is 2 times 4?"})

        def find_span(op_name):
            for call in mocked_upload_span.call_args_list:
                check_op_name = find_tag(call[0][0], 'operation')
                if check_op_name == op_name:
                    return call[0][0]

        llm_span = find_span('langchain_community.llms.fake.FakeListLLM')
        self.assertEqual(find_tag(llm_span, 'model_type'), 'chat')

    @patch.object(Uploader, 'upload_span')
    @patch('graphsignal.callbacks.langchain.v1.uuid_sha1', return_value='s1')
    async def test_chain_async_with_decorator(self, mocked_uuid_sha1, mocked_upload_span):
        prompt = PromptTemplate(
            input_variables=["product"],
            template="What is a good name for a company that makes {product}?",
        )

        llm = FakeListLLM(responses=['Final Answer:42'])
        chain = LLMChain(llm=llm, prompt=prompt)

        @graphsignal.trace_function
        async def run_chain():
            graphsignal.set_context_tag('session_id', 's2')
            await chain.ainvoke("colorful socks")

        await run_chain()

        def find_span(op_name):
            for call in mocked_upload_span.call_args_list:
                check_op_name = find_tag(call[0][0], 'operation')
                if check_op_name == op_name:
                    return call[0][0]

        run_chain_span = find_span('run_chain')
        self.assertIsNotNone(run_chain_span)

        llm_span = find_span('langchain_community.llms.fake.FakeListLLM')
        self.assertEqual(find_tag(llm_span, 'library'), 'langchain')
        self.assertEqual(find_tag(llm_span, 'session_id'), 's2')
