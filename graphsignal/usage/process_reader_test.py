import unittest
import logging
import sys
from unittest.mock import patch, Mock
import torch
from google.protobuf.json_format import MessageToJson
import pprint
import time

import graphsignal
from graphsignal.usage.process_reader import ProcessReader
from graphsignal.proto import profiles_pb2

logger = logging.getLogger('graphsignal')


class ProcessReaderTest(unittest.TestCase):
    def setUp(self):
        if len(logger.handlers) == 0:
            logger.addHandler(logging.StreamHandler(sys.stdout))
        graphsignal.configure(
            api_key='k1',
            workload_name='w1',
            debug_mode=True)

    def tearDown(self):
        graphsignal.shutdown()

    def test_read(self):
        profile = profiles_pb2.MLProfile()
        reader = graphsignal._agent.process_reader
        reader.read(profile)
        ProcessReader.MIN_CPU_READ_INTERVAL = 0
        time.sleep(0.2)
        reader.read(profile)

        #pp = pprint.PrettyPrinter()
        # pp.pprint(MessageToJson(profile))

        self.assertNotEqual(profile.node_usage.hostname, '')
        self.assertNotEqual(profile.node_usage.ip_address, '')
        self.assertNotEqual(profile.process_usage.process_id, '')
        if sys.platform != 'win32':
            self.assertTrue(profile.node_usage.mem_total > 0)
            self.assertTrue(profile.node_usage.mem_used > 0)
            self.assertTrue(profile.process_usage.cpu_usage_percent > 0)
            self.assertTrue(profile.process_usage.max_rss > 0)
            self.assertTrue(profile.process_usage.current_rss > 0)
            self.assertTrue(profile.process_usage.vm_size > 0)
