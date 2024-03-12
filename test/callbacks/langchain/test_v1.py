import unittest
import logging
import sys
import os
from unittest.mock import patch, Mock
from google.protobuf.json_format import MessageToJson
import pprint
import openai
from langchain.agents import Tool, initialize_agent, load_tools
from langchain.llms import OpenAI
from typing import Any, List, Mapping, Optional
from langchain.llms.base import LLM
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.callbacks.base import CallbackManager

import graphsignal
from graphsignal.uploader import Uploader
from graphsignal.recorders.openai_recorder import OpenAIRecorder
from graphsignal.callbacks.langchain.v1 import GraphsignalCallbackHandler
from test.proto_utils import find_tag, find_usage

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

    @patch.object(Uploader, 'upload_span')
    async def test_chain(self, mocked_upload_span):
        llm = DummyLLM()
        tools = load_tools(["llm-math"], llm=llm)
        agent = initialize_agent(
            tools, llm, agent="zero-shot-react-description", verbose=True)
        agent.run("What is 2 raised to .123243 power?")

        t3 = mocked_upload_span.call_args_list[0][0][0]
        t2 = mocked_upload_span.call_args_list[1][0][0]
        t1 = mocked_upload_span.call_args_list[2][0][0]

        self.assertEqual(find_tag(t1, 'operation'), 'langchain.chains.AgentExecutor')
        self.assertEqual(find_usage(t1, 'inputs', 'byte_count'), 34.0)
        self.assertEqual(find_usage(t1, 'outputs', 'byte_count'), 2.0)

        self.assertEqual(find_tag(t2, 'operation'), 'langchain.chains.LLMChain')
        self.assertEqual(t2.context.parent_span_id, t1.span_id)
        self.assertEqual(t2.context.root_span_id, t1.span_id)
        self.assertEqual(find_usage(t2, 'inputs', 'byte_count'), 61.0)
        self.assertEqual(find_usage(t2, 'outputs', 'byte_count'), 15.0)

        self.assertEqual(find_tag(t3, 'operation'), 'langchain.llms.DummyLLM')
        self.assertEqual(t3.context.parent_span_id, t2.span_id)
        self.assertEqual(t3.context.root_span_id, t1.span_id)
