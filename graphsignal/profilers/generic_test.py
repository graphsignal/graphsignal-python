import unittest
import logging
import sys
import time
from unittest.mock import patch, Mock
from google.protobuf.json_format import MessageToJson
import pprint

import graphsignal
from graphsignal.profilers.generic import profile_step
from graphsignal.proto import profiles_pb2
from graphsignal.uploader import Uploader

logger = logging.getLogger('graphsignal')


class GenericProfilerTest(unittest.TestCase):
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
    def test_profile_step(self, mocked_upload_profile):
        with profile_step(phase_name='training', ensure_profile=True):
            time.sleep(0.1)

        profile = mocked_upload_profile.call_args[0][0]

        #pp = pprint.PrettyPrinter()
        #pp.pprint(MessageToJson(profile))

        self.assertEqual(profile.phase_name, 'training')
        self.assertTrue(profile.step_stats.step_count > 0)
        self.assertTrue(profile.step_stats.total_time_us > 0)
