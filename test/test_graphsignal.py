import unittest
import logging
import sys
import os
from unittest.mock import patch, Mock

import graphsignal
from graphsignal.spans import Span
from graphsignal.uploader import Uploader

logger = logging.getLogger('graphsignal')


class GraphsignalTest(unittest.TestCase):
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
        arg1 = graphsignal._read_config_param('arg1', str, 'val1', required=True)
        self.assertEqual(arg1, 'val1')

        arg2 = graphsignal._read_config_param('arg2', int, 1, required=True)
        self.assertEqual(arg2, 1)

        arg3 = graphsignal._read_config_param('arg3', int, None, required=False)
        self.assertEqual(arg3, None)

        os.environ['GRAPHSIGNAL_ARG4'] = '2'
        arg4 = graphsignal._read_config_param('arg4', int, None, required=False)
        self.assertEqual(arg4, 2)

        with self.assertRaises(ValueError):
            arg5 = graphsignal._read_config_param('arg5', str, None, required=True)

        os.environ['GRAPHSIGNAL_ARG6'] = '10'
        arg6 = graphsignal._read_config_param('arg6', int, None, required=True)
        self.assertEqual(arg6, 10)

        os.environ['GRAPHSIGNAL_ARG7'] = 'str'
        with self.assertRaises(ValueError):
            arg7 = graphsignal._read_config_param('arg7', int, None, required=True)

        env_tags = graphsignal._read_config_tags({'arg8': 'v1', 'arg9': '2.0'})
        self.assertEqual(env_tags, {'arg8': 'v1', 'arg9': '2.0'})

        os.environ['GRAPHSIGNAL_TAG_ARG10'] = 'v1'
        os.environ['GRAPHSIGNAL_TAG_ARG11'] = '2.0'
        env_tags = graphsignal._read_config_tags(None)
        self.assertEqual(env_tags, {'arg10': 'v1', 'arg11': '2.0'})


    def test_configure(self):
        self.assertEqual(graphsignal._tracer.api_key, 'k1')
        self.assertEqual(graphsignal._tracer.debug_mode, True)

    @patch.object(Span, '_stop', return_value=None)
    @patch.object(Span, '_start', return_value=None)
    def test_trace_function(self, mocked_start, mocked_stop):
        @graphsignal.trace_function
        def test_func(p):
            return 1 + p

        ret = test_func(12)
        self.assertEqual(ret, 13)

        mocked_start.assert_called_once()
        mocked_stop.assert_called_once()

    @patch.object(Span, '_stop', return_value=None)
    @patch.object(Span, '_start', return_value=None)
    def test_trace_function_with_args(self, mocked_start, mocked_stop):
        @graphsignal.trace_function(operation='ep1', tags=dict(t1='v1'))
        def test_func(p):
            return 1 + p

        ret = test_func(12)
        self.assertEqual(ret, 13)

        mocked_start.assert_called_once()
        mocked_stop.assert_called_once()

    @patch.object(Uploader, 'upload_score')
    def test_score(self, mocked_upload_score):
        graphsignal.score(
            name='s1', 
            tags=dict(t1='v1'), 
            score=0.5, 
            unit='u1',
            severity=2, 
            comment='c1')

        model = mocked_upload_score.call_args[0][0]

        self.assertTrue(model.score_id is not None)
        self.assertEqual(model.name, 's1')
        self.assertEqual(find_tag(model, 't1'), 'v1')
        self.assertEqual(model.score, 0.5)
        self.assertEqual(model.unit, 'u1')
        self.assertEqual(model.severity, 2)
        self.assertEqual(model.comment, 'c1')
        self.assertTrue(model.create_ts > 0)

def find_tag(model, key):
    for tag in model.tags:
        if tag.key == key:
            return tag.value
    return None