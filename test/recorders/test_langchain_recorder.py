import unittest
import logging
import sys
from unittest.mock import patch, Mock
import pprint
from typing import Any, List, Mapping, Optional
from langchain.llms.base import LLM


import graphsignal
from graphsignal import client
from graphsignal.uploader import Uploader
from graphsignal.recorders.langchain_recorder import LangChainRecorder
from graphsignal.recorders.openai_recorder import OpenAIRecorder

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

    async def test_record(self):
        recorder = LangChainRecorder()
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

        self.assertEqual(model.config[0].key, 'langchain.library.version')

    @patch.object(Uploader, 'upload_span')
    async def test_chain(self, mocked_upload_span):
        llm = DummyLLM()

        llm.generate(["test"])

        t1 = mocked_upload_span.call_args_list[0][0][0]
