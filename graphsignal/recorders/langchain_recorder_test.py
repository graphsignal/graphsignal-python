import unittest
import logging
import sys
import os
import json
import time
from unittest.mock import patch, Mock
from google.protobuf.json_format import MessageToJson
import pprint
import langchain
from langchain.agents import Tool, initialize_agent, load_tools
from langchain.llms import OpenAI
from typing import Any, List, Mapping, Optional
from langchain.llms.base import LLM

import graphsignal
from graphsignal.proto import signals_pb2
from graphsignal.uploader import Uploader
from graphsignal.traces import DEFAULT_OPTIONS
from graphsignal.recorders.langchain_recorder import LangChainRecorder

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
        os.environ['OPENAI_API_KEY'] = 'dummy-api-key'
        llm = OpenAI(temperature=0)
        llm = DummyLLM()
        tools = load_tools(["llm-math"], llm=llm)
        agent = initialize_agent(
            tools, llm, agent="zero-shot-react-description", verbose=True
        )
        agent.run("What is 2 raised to .123243 power?")

        proto = mocked_upload_trace.call_args[0][0]

        #pp = pprint.PrettyPrinter()
        #pp.pprint(MessageToJson(proto))

        self.assertEqual(proto.frameworks[0].name, 'LangChain')

        self.assertEqual(find_data_count(proto, 'inputs', 'byte_count'), 34.0)
        self.assertEqual(find_data_count(proto, 'inputs', 'element_count'), 1.0)
        self.assertEqual(find_data_count(proto, 'outputs', 'byte_count'), 2.0)
        self.assertEqual(find_data_count(proto, 'outputs', 'element_count'), 1.0)

        self.assertEqual(proto.root_span.spans[0].name, 'langchain.chains.LLMChain')
        self.assertTrue(proto.root_span.spans[0].start_ns > 0)
        self.assertTrue(proto.root_span.spans[0].end_ns > 0)
        self.assertTrue(proto.root_span.spans[0].is_endpoint)
        self.assertEqual(proto.root_span.spans[0].spans[0].name, 'langchain.llms.DummyLLM')
        self.assertTrue(proto.root_span.spans[0].spans[0].start_ns > 0)
        self.assertTrue(proto.root_span.spans[0].spans[0].end_ns > 0)
        self.assertTrue(proto.root_span.spans[0].spans[0].is_endpoint)


def find_data_count(proto, data_name, count_name):
    for data_stats in proto.data_profile:
        if data_stats.data_name == data_name:
            for data_count in data_stats.counts:
                if data_count.name == count_name:
                    return data_count.count
    return None