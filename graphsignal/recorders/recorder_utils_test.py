
import unittest
import logging
import sys
import os
import json
import time
from unittest.mock import patch, Mock
from google.protobuf.json_format import MessageToJson
import pprint

import graphsignal
from graphsignal.recorders.recorder_utils import patch_method

logger = logging.getLogger('graphsignal')


class Test:
    def __init__(self):
        pass

    def test(self, a, b, c=None):
        return a + 1

    def test_exc(self):
        raise Exception('exc1')


class RecorderUtilsTest(unittest.TestCase):
    def setUp(self):
        if len(logger.handlers) == 0:
            logger.addHandler(logging.StreamHandler(sys.stdout))
        graphsignal.configure(
            api_key='k1',
            deployment='d1',
            debug_mode=True)

    def tearDown(self):
        graphsignal.shutdown()

    def test_patch_method(self):
        obj = Test()

        before_func_called = False
        def before_func(args, kwargs):
            nonlocal before_func_called
            before_func_called = True
            self.assertEqual(args, (1, 2))
            self.assertEqual(kwargs, {'c': 3})
            return dict(d=1)

        after_func_called = False
        def after_func(args, kwargs, ret, exc, context):
            nonlocal after_func_called
            after_func_called = True
            self.assertEqual(args, (1, 2))
            self.assertEqual(kwargs, {'c': 3})
            self.assertEqual(ret, 2)
            self.assertIsNone(exc)
            self.assertEqual(context['d'], 1)

        self.assertTrue(patch_method(obj, 'test', before_func=before_func, after_func=after_func))

        obj.test(1, 2, c=3)

        self.assertTrue(before_func_called)
        self.assertTrue(after_func_called)

    def test_patch_method_exc(self):
        obj = Test()

        after_func_called = False
        def after_func(args, kwargs, ret, exc, context):
            nonlocal after_func_called
            after_func_called = True
            self.assertEqual(str(exc), 'exc1')

        self.assertTrue(patch_method(obj, 'test_exc', after_func=after_func))

        with self.assertRaises(Exception) as context:
            obj.test_exc()

        self.assertTrue(after_func_called)