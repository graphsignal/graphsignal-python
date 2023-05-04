import unittest
import logging
import sys
from unittest.mock import patch, Mock
from google.protobuf.json_format import MessageToJson
import pprint
from typing import Any, List, Mapping, Optional
from langchain.llms.base import LLM


import graphsignal
from graphsignal.proto import signals_pb2
from graphsignal.uploader import Uploader
from graphsignal.traces import DEFAULT_OPTIONS
from graphsignal.recorders.langchain_recorder import LangChainRecorder
from graphsignal.recorders.openai_recorder import OpenAIRecorder

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
        llm = DummyLLM()

        llm.generate(["test"])

        t1 = mocked_upload_trace.call_args_list[0][0][0]

        self.assertEqual(t1.frameworks[0].name, 'OpenAI Python Library')
        self.assertEqual(t1.frameworks[1].name, 'LangChain')
