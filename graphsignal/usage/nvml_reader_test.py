import unittest
import logging
import sys
from unittest.mock import patch, Mock
import torch
from google.protobuf.json_format import MessageToJson
import pprint
import time

import graphsignal
from graphsignal.usage.nvml_reader import NvmlReader
from graphsignal.proto import profiles_pb2

logger = logging.getLogger('graphsignal')


class NvmlReaderTest(unittest.TestCase):
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
        reader = graphsignal._agent.nvml_reader
        reader.read(resource_usage)

        #pp = pprint.PrettyPrinter()
        # pp.pprint(MessageToJson(resource_usage))

        if len(resource_usage.device_usage) > 0:
            device_usage = resource_usage.device_usage[0]
            self.assertNotEqual(device_usage.device_id, '')
            self.assertNotEqual(device_usage.device_name, '')
            self.assertTrue(device_usage.mem_total > 0)
            self.assertTrue(device_usage.mem_used > 0)
            self.assertTrue(device_usage.mem_free > 0)
            #self.assertTrue(device_usage.gpu_utilization_percent > 0)
            #self.assertTrue(device_usage.mem_utilization_percent > 0)
            self.assertTrue(device_usage.gpu_temp_c > 0)
            self.assertTrue(device_usage.power_usage_w > 0)
            self.assertTrue(device_usage.fan_speed_percent > 0)
