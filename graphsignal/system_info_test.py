import unittest
import logging
import sys
from unittest.mock import patch, Mock

import graphsignal
from graphsignal.proto import profiles_pb2
from graphsignal.system_info import compare_semver

logger = logging.getLogger('graphsignal')


class SystemInfoTest(unittest.TestCase):
    def setUp(self):
        graphsignal.configure(
            api_key='k1',
            workload_name='w1',
            debug_mode=True)

    def tearDown(self):
        graphsignal.shutdown()

    def test_compare_semver(self):
        s1 = profiles_pb2.SemVer()

        s1.major = 1
        s1.minor = 2
        self.assertEqual(compare_semver(s1, (1, 3, 0)), -1)

        s1.major = 1
        s1.minor = 2
        s1.patch = 3
        self.assertEqual(compare_semver(s1, (1, 2, 3)), 0)

        s1.major = 1
        s1.minor = 2
        s1.patch = 3
        self.assertEqual(compare_semver(s1, (1, 2, 2)), 1)
