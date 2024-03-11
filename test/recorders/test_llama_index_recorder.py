import unittest
import logging
import sys
from unittest.mock import patch, Mock
from google.protobuf.json_format import MessageToJson
import pprint
from typing import Any, List, Mapping, Optional


import graphsignal
from graphsignal.proto import signals_pb2
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
        proto = signals_pb2.Span()
        context = {}
        recorder.on_span_start(proto, context)
        recorder.on_span_stop(proto, context)
        recorder.on_span_read(proto, context)

        #self.assertEqual(proto.config[0].key, 'llama_index.library.version')
