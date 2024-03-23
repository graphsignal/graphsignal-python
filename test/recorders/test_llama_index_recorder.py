import unittest
import logging
import sys
from unittest.mock import patch, Mock
import pprint
from typing import Any, List, Mapping, Optional


import graphsignal
from graphsignal import client
from graphsignal.recorders.llama_index_recorder import LlamaIndexRecorder
from graphsignal.recorders.openai_recorder import OpenAIRecorder

logger = logging.getLogger('graphsignal')


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
        recorder = LlamaIndexRecorder()
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

        #self.assertEqual(model.config[0].key, 'llama_index.library.version')
