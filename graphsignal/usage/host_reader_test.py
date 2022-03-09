import unittest
import logging
import sys
from unittest.mock import patch, Mock
import torch
from google.protobuf.json_format import MessageToJson
import pprint
import time

import graphsignal
from graphsignal.usage.host_reader import HostReader
from graphsignal.proto import profiles_pb2

logger = logging.getLogger('graphsignal')


class HostReaderTest(unittest.TestCase):
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
        resource_usage = profiles_pb2.ResourceUsage()
        reader = graphsignal._agent.host_reader
        reader.read(resource_usage)
        HostReader.MIN_CPU_READ_INTERVAL = 0
        time.sleep(0.2)
        reader.read(resource_usage)

        #pp = pprint.PrettyPrinter()
        # pp.pprint(MessageToJson(resource_usage))

        self.assertTrue(resource_usage.host_usage.cpu_usage_percent > 0)
        self.assertTrue(resource_usage.host_usage.max_rss > 0)
        self.assertTrue(resource_usage.host_usage.current_rss > 0)
        self.assertTrue(resource_usage.host_usage.vm_size > 0)
