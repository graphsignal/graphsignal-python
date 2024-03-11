
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
from graphsignal.recorders.instrumentation import patch_method, instrument_method, read_args, parse_semver, compare_semver
from graphsignal.uploader import Uploader

logger = logging.getLogger('graphsignal')


class Dummy:
    def __init__(self):
        pass

    def test(self, a, b, c=None):
        return a + 1

    def test_exc(self):
        raise Exception('exc1')

    def test_gen(self):
        for i in range(2):
            yield 'item' + str(i)


class InstrumentationTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        if len(logger.handlers) == 0:
            logger.addHandler(logging.StreamHandler(sys.stdout))
        graphsignal.configure(
            api_key='k1',
            deployment='d1',
            upload_on_shutdown=False,
            debug_mode=True)

    async def asyncTearDown(self):
        graphsignal.shutdown()

    @patch.object(Uploader, 'upload_span')
    async def test_instrument_method(self, mocked_upload_span):
        obj = Dummy()

        trace_func_called = False
        def trace_func(span, args, kwargs, ret, exc):
            nonlocal trace_func_called
            trace_func_called = True

        instrument_method(obj, 'test', 'ep1', trace_func=trace_func)

        obj.test(1, 2, c=3)

        proto = mocked_upload_span.call_args[0][0]

        self.assertTrue(trace_func_called)
        self.assertEqual(proto.tags[1].value, 'ep1')

    @patch.object(Uploader, 'upload_span')
    async def test_instrument_method_generator(self, mocked_upload_span):
        obj = Dummy()

        trace_func_called = None
        def trace_func(span, args, kwargs, ret, exc):
            nonlocal trace_func_called
            trace_func_called = True

        instrument_method(obj, 'test_gen', 'ep1', trace_func=trace_func)

        for item in obj.test_gen():
            pass

        proto = mocked_upload_span.call_args[0][0]

        self.assertTrue(trace_func_called)
        self.assertEqual(proto.tags[1].value, 'ep1')
        self.assertTrue(proto.context.start_ns > 0)
        self.assertTrue(proto.context.end_ns > 0)
        self.assertTrue(proto.context.first_token_ns > 0)

    async def test_patch_method(self):
        obj = Dummy()

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

    async def test_patch_method_exc(self):
        obj = Dummy()

        after_func_called = False
        def after_func(args, kwargs, ret, exc, context):
            nonlocal after_func_called
            after_func_called = True
            self.assertEqual(str(exc), 'exc1')

        self.assertTrue(patch_method(obj, 'test_exc', after_func=after_func))

        with self.assertRaises(Exception) as context:
            obj.test_exc()

        self.assertTrue(after_func_called)

    async def test_patch_method_generator(self):
        obj = Dummy()

        yield_func_called = False
        def yield_func(stopped, item, context):
            nonlocal yield_func_called
            yield_func_called = True
            if not stopped:
                self.assertTrue(item in ('item0', 'item1'))

        self.assertTrue(patch_method(obj, 'test_gen', yield_func=yield_func))

        for item in obj.test_gen():
            pass

        self.assertTrue(yield_func_called)

    async def test_read_args(self):
        def test(*args, **kwargs):
            values = read_args(args, kwargs, ['a', 'b', 'c'])
            self.assertEqual(values, {'a': 1, 'b': 2, 'c': 3})

        test(1, 2, c=3)

    async def test_parse_semver(self):
        parsed_version = parse_semver('1.2.3')
        self.assertEqual(parsed_version[0], 1)
        self.assertEqual(parsed_version[1], 2)
        self.assertEqual(parsed_version[2], 3)

        parsed_version = parse_semver('1.2')
        self.assertEqual(parsed_version[0], 1)
        self.assertEqual(parsed_version[1], 2)
        self.assertEqual(parsed_version[2], 0)

    async def test_compare_semver(self):
        self.assertEqual(compare_semver((1, 2, 0), (1, 3, 0)), -1)

        self.assertEqual(compare_semver((1, 2, 3), (1, 2, 3)), 0)

        self.assertEqual(compare_semver((1, 2, 3), (1, 2, 2)), 1)