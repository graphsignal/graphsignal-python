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

import graphsignal
from graphsignal.proto import signals_pb2
from graphsignal.uploader import Uploader
from graphsignal.spans import DEFAULT_OPTIONS
from graphsignal.recorders.autogpt_recorder import AutoGPTRecorder

logger = logging.getLogger('graphsignal')


class AutoGPTRecorderTest(unittest.IsolatedAsyncioTestCase):
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
    @patch.object(openai.ChatCompletion, 'create')
    async def test_trace_run(self, mocked_run, mocked_upload_span):
        # TODO: automate testing of autogpt
        pass

        #if os.path.isdir('autogpt'):
        #    sys.path.append(os.getcwd())
        #    import autogpt
        #    autogpt.__main__.main()
