import unittest
import logging
import sys
from unittest.mock import patch, Mock

import graphsignal
from graphsignal.proto import profiles_pb2
from graphsignal.proto_utils import parse_semver, compare_semver

logger = logging.getLogger('graphsignal')


class SystemInfoTest(unittest.TestCase):
    def setUp(self):
        graphsignal.configure(
            api_key='k1',
            workload_name='w1',
            debug_mode=True)

    def tearDown(self):
        graphsignal.shutdown()

    def test_parse_semver(self):
        s1 = profiles_pb2.SemVer()
        parse_semver(s1, '1.2.3')
        self.assertEqual(s1.major, 1)
        self.assertEqual(s1.minor, 2)
        self.assertEqual(s1.patch, 3)

        s2 = profiles_pb2.SemVer()
        parse_semver(s2, '1.2')
        self.assertEqual(s2.major, 1)
        self.assertEqual(s2.minor, 2)
        self.assertEqual(s2.patch, 0)

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
