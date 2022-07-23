import unittest
import logging
import sys
from unittest.mock import patch, Mock
from google.protobuf.json_format import MessageToJson
import pprint

import graphsignal
from graphsignal.proto import profiles_pb2
from graphsignal.uploader import Uploader

logger = logging.getLogger('graphsignal')


class HuggingFaceGeneratorTest(unittest.TestCase):
    def setUp(self):
        if len(logger.handlers) == 0:
            logger.addHandler(logging.StreamHandler(sys.stdout))
        graphsignal.configure(
            api_key='k1',
            workload_name='w1',
            debug_mode=True)

    def tearDown(self):
        graphsignal.shutdown()

    @patch.object(Uploader, 'upload_profile')
    def test_generator(self, mocked_upload_profile):
        from transformers import pipeline
        from graphsignal.profilers.pytorch import profile_inference

        generator = pipeline(task="text-generation", model='distilgpt2')

        with profile_inference():
            output = generator('some text')

        graphsignal.upload()
        profile = mocked_upload_profile.call_args[0][0]

        #pp = pprint.PrettyPrinter()
        #pp.pprint(MessageToJson(profile))

        test_op_stats = None
        for op_stats in profile.op_stats:
            if op_stats.op_name == 'aten::mm':
                test_op_stats = op_stats
                break
        self.assertIsNotNone(test_op_stats)
        self.assertTrue(test_op_stats.count >= 1)
        self.assertTrue(test_op_stats.total_host_time_us >= 1)
        self.assertTrue(test_op_stats.self_host_time_us >= 1)
