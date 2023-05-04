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

@unittest.skip("langchain <=0.0.153")
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

    @patch.object(Uploader, 'upload_trace')
    async def test_chain(self, mocked_upload_trace):
        llm = DummyLLM()
        tools = load_tools(["llm-math"], llm=llm)
        agent = initialize_agent(
            tools, llm, agent="zero-shot-react-description", verbose=True)
        agent.run("What is 2 raised to .123243 power?")

        t3 = mocked_upload_trace.call_args_list[0][0][0]
        t2 = mocked_upload_trace.call_args_list[1][0][0]
        t1 = mocked_upload_trace.call_args_list[2][0][0]

        #pp = pprint.PrettyPrinter()
        #pp.pprint(MessageToJson(t1))
        #pp.pprint(MessageToJson(t2))
        #pp.pprint(MessageToJson(t3))

        self.assertEqual(t1.labels, ['root'])
        self.assertEqual(find_tag(t1, 'component'), 'Agent')
        self.assertEqual(find_tag(t1, 'operation'), 'langchain.chains.AgentExecutor')
        self.assertEqual(find_data_count(t1, 'inputs', 'byte_count'), 34.0)
        self.assertEqual(find_data_count(t1, 'inputs', 'element_count'), 1.0)
        self.assertEqual(find_data_count(t1, 'outputs', 'byte_count'), 2.0)
        self.assertEqual(find_data_count(t1, 'outputs', 'element_count'), 1.0)

        self.assertEqual(t2.labels, [])
        self.assertEqual(find_tag(t2, 'operation'), 'langchain.chains.LLMChain')
        self.assertEqual(t2.span.parent_trace_id, t1.trace_id)
        self.assertEqual(t2.span.root_trace_id, t1.trace_id)
        self.assertEqual(find_data_count(t2, 'inputs', 'byte_count'), 61.0)
        self.assertEqual(find_data_count(t2, 'inputs', 'element_count'), 4.0)
        self.assertEqual(find_data_count(t2, 'outputs', 'byte_count'), 15.0)
        self.assertEqual(find_data_count(t2, 'outputs', 'element_count'), 1.0)

        self.assertEqual(t3.labels, [])
        self.assertEqual(find_tag(t3, 'component'), 'LLM')
        self.assertEqual(find_tag(t3, 'operation'), 'langchain.llms.DummyLLM')
        self.assertEqual(t3.span.parent_trace_id, t2.trace_id)
        self.assertEqual(t3.span.root_trace_id, t1.trace_id)


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