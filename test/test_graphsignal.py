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
        graphsignal._tracer.auto_export = False

    def tearDown(self):
        graphsignal.shutdown()

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
