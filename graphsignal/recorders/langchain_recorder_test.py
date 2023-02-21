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
from graphsignal.endpoint_trace import DEFAULT_OPTIONS
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
            debug_mode=True)

    async def asyncTearDown(self):
        graphsignal.shutdown()

    async def test_record(self):
        recorder = LangChainRecorder()
        recorder.setup()
        signal = signals_pb2.WorkerSignal()
        context = {}
        recorder.on_trace_start(signal, context, DEFAULT_OPTIONS)
        recorder.on_trace_stop(signal, context, DEFAULT_OPTIONS)
        recorder.on_trace_read(signal, context, DEFAULT_OPTIONS)

        self.assertEqual(signal.frameworks[0].name, 'LangChain')

    @patch.object(Uploader, 'upload_signal')
    async def test_chain(self, mocked_upload_signal):
        llm = OpenAI(temperature=0)
        llm = DummyLLM()
        tools = load_tools(["llm-math"], llm=llm)
        agent = initialize_agent(
            tools, llm, agent="zero-shot-react-description", verbose=True
        )
        agent.run("What is 2 raised to .123243 power?")

        signal = mocked_upload_signal.call_args[0][0]

        #pp = pprint.PrettyPrinter()
        #pp.pprint(MessageToJson(signal))

        self.assertEqual(signal.frameworks[0].name, 'LangChain')

        self.assertEqual(find_data_metric(signal, 'inputs', 'byte_count'), 34.0)
        self.assertEqual(find_data_metric(signal, 'inputs', 'element_count'), 1.0)
        self.assertEqual(find_data_metric(signal, 'outputs', 'byte_count'), 2.0)
        self.assertEqual(find_data_metric(signal, 'outputs', 'element_count'), 1.0)

        self.assertEqual(signal.root_span.spans[0].name, 'langchain.chains.LLMChain')
        self.assertTrue(signal.root_span.spans[0].start_ns > 0)
        self.assertTrue(signal.root_span.spans[0].end_ns > 0)
        self.assertTrue(signal.root_span.spans[0].is_endpoint)
        self.assertEqual(signal.root_span.spans[0].spans[0].name, 'langchain.llms.DummyLLM')
        self.assertTrue(signal.root_span.spans[0].spans[0].start_ns > 0)
        self.assertTrue(signal.root_span.spans[0].spans[0].end_ns > 0)
        self.assertTrue(signal.root_span.spans[0].spans[0].is_endpoint)


def find_data_metric(signal, data_name, metric_name):
    for data_metric in signal.data_metrics:
        if data_metric.data_name == data_name and data_metric.metric_name == metric_name:
            return data_metric.metric.gauge
    return None