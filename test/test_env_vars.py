import unittest
import logging
import sys
import os
from unittest.mock import patch, Mock

import graphsignal
from graphsignal.env_vars import read_config_param, read_config_tags

logger = logging.getLogger('graphsignal')


class EnvVarsTest(unittest.TestCase):
    def setUp(self):
        if len(logger.handlers) == 0:
            logger.addHandler(logging.StreamHandler(sys.stdout))
        graphsignal.configure(
            api_key='k1',
            debug_mode=True)
        graphsignal._tracer.export_on_shutdown = False


    def tearDown(self):
        graphsignal.shutdown()

    def test_read_config_test(self):
        arg1 = read_config_param('arg1', str, 'val1', required=True)
        self.assertEqual(arg1, 'val1')

        arg2 = read_config_param('arg2', int, 1, required=True)
        self.assertEqual(arg2, 1)

        arg3 = read_config_param('arg3', int, None, required=False)
        self.assertEqual(arg3, None)

        os.environ['GRAPHSIGNAL_ARG4'] = '2'
        arg4 = read_config_param('arg4', int, None, required=False)
        self.assertEqual(arg4, 2)

        with self.assertRaises(ValueError):
            arg5 = read_config_param('arg5', str, None, required=True)

        os.environ['GRAPHSIGNAL_ARG6'] = '10'
        arg6 = read_config_param('arg6', int, None, required=True)
        self.assertEqual(arg6, 10)

        os.environ['GRAPHSIGNAL_ARG7'] = 'str'
        with self.assertRaises(ValueError):
            arg7 = read_config_param('arg7', int, None, required=True)

        env_tags = read_config_tags({'arg8': 'v1', 'arg9': '2.0'})
        self.assertEqual(env_tags, {'arg8': 'v1', 'arg9': '2.0'})

        os.environ['GRAPHSIGNAL_ARG9'] = 'v1,v2'
        arg9 = read_config_param('arg9', list, None)
        self.assertEqual(arg9, ['v1', 'v2'])

        os.environ['GRAPHSIGNAL_TAG_ARG10'] = 'v1'
        os.environ['GRAPHSIGNAL_TAG_ARG11'] = '2.0'
        env_tags = read_config_tags(None)
        self.assertEqual(env_tags, {'arg10': 'v1', 'arg11': '2.0'})
