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
from graphsignal.proto import signals_pb2

logger = logging.getLogger('graphsignal')


class ProcessReaderTest(unittest.TestCase):
    def setUp(self):
        if len(logger.handlers) == 0:
            logger.addHandler(logging.StreamHandler(sys.stdout))
        graphsignal.configure(
            api_key='k1',
            debug_mode=True)

    def tearDown(self):
        graphsignal.shutdown()

    def test_read(self):
        signal = signals_pb2.MLSignal()
        reader = graphsignal._agent.process_reader
        reader.read(signal)
        ProcessReader.MIN_CPU_READ_INTERVAL_NS = 0
        time.sleep(0.2)
        reader.read(signal)

        #pp = pprint.PrettyPrinter()
        # pp.pprint(MessageToJson(signal))

        self.assertNotEqual(signal.node_usage.hostname, '')
        self.assertNotEqual(signal.node_usage.ip_address, '')
        self.assertNotEqual(signal.process_usage.process_id, '')
        self.assertTrue(signal.process_usage.start_ms > 0)
        if sys.platform != 'win32':
            self.assertTrue(signal.node_usage.mem_total > 0)
            self.assertTrue(signal.node_usage.mem_used > 0)
            self.assertTrue(signal.process_usage.cpu_usage_percent > 0)
            self.assertTrue(signal.process_usage.max_rss > 0)
            self.assertTrue(signal.process_usage.current_rss > 0)
            self.assertTrue(signal.process_usage.vm_size > 0)
        self.assertTrue(signal.agent_info.version.major > 0 or signal.agent_info.version.minor > 0)
