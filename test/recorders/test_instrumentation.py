
import unittest
import logging
import sys
import os
import json
import time
import asyncio
from unittest.mock import patch, Mock
import pprint

import graphsignal
from graphsignal.recorders.instrumentation import patch_method, trace_method, profile_method, read_args, parse_semver, compare_semver
from graphsignal.spans import Span
from graphsignal.uploader import Uploader
from test.model_utils import find_tag, find_counter

logger = logging.getLogger('graphsignal')


class Dummy:
    def __init__(self):
        pass

    def test(self, a, b, c=None):
        time.sleep(0.001)
        return a + 1

    def test_exc(self):
        raise Exception('exc1')

    def test_gen(self):
        time.sleep(0.001)
        for i in range(2):
            yield 'item' + str(i)

    async def test_gen_async(self):
        for i in range(2):
            await asyncio.sleep(0.001)
            yield 'item' + str(i)

class DummyNoSleep:
    def __init__(self):
        pass

    def test(self, a, b, c=None):
        return a + 1

    def test_exc(self):
        raise Exception('exc1')

    def test_gen(self):
        for i in range(2):
            yield 'item' + str(i)

    async def test_gen_async(self):
        for i in range(2):
            yield 'item' + str(i)

class InstrumentationTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        if len(logger.handlers) == 0:
            logger.addHandler(logging.StreamHandler(sys.stdout))
        graphsignal.configure(
            api_key='k1',
            debug_mode=True)
        graphsignal._tracer.auto_export = False

    async def asyncTearDown(self):
        graphsignal.shutdown()

    async def test_profile_method(self):
        obj = Dummy()

        def event_name_func(args, kwargs):
            return 'event2'

        measured_duration_ns = 0
        def profile_func(event_name, duration_ns):
            self.assertEqual(event_name, 'event2')
            self.assertTrue(duration_ns > 0)
            nonlocal measured_duration_ns
            measured_duration_ns += duration_ns

        profile_method(
            obj=obj, 
            func_name='test', 
            event_name_func=event_name_func, 
            profile_func=profile_func)

        obj.test(1, 2, c=3)
        obj.test(1, 2, c=3)
        obj.test(1, 2, c=3)

        self.assertTrue(measured_duration_ns > 0)

    async def test_profile_method_generator(self):
        obj = Dummy()

        measured_duration_ns = 0
        def profile_func(event_name, duration_ns):
            self.assertEqual(event_name, 'event1')
            self.assertTrue(duration_ns > 0)
            nonlocal measured_duration_ns
            measured_duration_ns += duration_ns

        profile_method(
            obj=obj, 
            func_name='test_gen', 
            event_name='event1', 
            profile_func=profile_func)

        for item in obj.test_gen():
            pass

        self.assertTrue(measured_duration_ns > 0)


    @patch.object(Uploader, 'upload_span')
    async def test_trace_method(self, mocked_upload_span):
        obj = Dummy()

        trace_func_called = False
        def trace_func(span, args, kwargs, ret, exc):
            span.set_sampled(True)
            nonlocal trace_func_called
            trace_func_called = True

        trace_method(obj, 'test', 'op1', trace_func=trace_func)

        obj.test(1, 2, c=3)

        model = mocked_upload_span.call_args[0][0]

        self.assertTrue(trace_func_called)
        self.assertEqual(model.name, 'op1')

    @patch.object(Uploader, 'upload_span')
    async def test_trace_method_generator(self, mocked_upload_span):
        obj = Dummy()

        trace_func_called = None
        def trace_func(span, args, kwargs, ret, exc):
            span.set_sampled(True)
            nonlocal trace_func_called
            trace_func_called = True

        data_func_called = False
        def data_func(span, item):
            nonlocal data_func_called
            data_func_called = True

        trace_method(obj, 'test_gen', 'op1', trace_func=trace_func, data_func=data_func)

        for item in obj.test_gen():
            pass

        model = mocked_upload_span.call_args[0][0]

        self.assertTrue(trace_func_called)
        self.assertTrue(data_func_called)
        self.assertEqual(model.name,'op1')
        self.assertTrue(find_counter(model, 'span.duration') > 0)

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

    async def test_patch_method_async_generator(self):
        obj = Dummy()

        yield_func_called = False
        yield_func_stop_called = False
        def yield_func(stopped, item, context):
            nonlocal yield_func_called, yield_func_stop_called
            if not stopped:
                yield_func_called = True
                self.assertTrue(item in ('item0', 'item1'))
            else:
                yield_func_stop_called = True

        self.assertTrue(patch_method(obj, 'test_gen_async', yield_func=yield_func))

        async for item in obj.test_gen_async():
            pass

        self.assertTrue(yield_func_called)
        self.assertTrue(yield_func_stop_called)

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

    #@unittest.skip('for now')
    @patch.object(Uploader, 'upload_span')
    def test_overhead(self, mocked_upload_span):
        #import cProfile, pstats
        #profiler = cProfile.Profile()
        #profiler.enable()

        graphsignal._tracer.debug_mode = False
        logger.setLevel(logging.ERROR)

        obj = DummyNoSleep()

        def trace_func(span, args, kwargs, ret, exc):
            span.set_sampled(True)
        trace_method(obj, 'test', 'op1', trace_func=trace_func)

        calls = 1000
        start_ns = time.perf_counter_ns()
        for _ in range(calls):
            obj.test(1, 2, c=3)
        took_ns = time.perf_counter_ns() - start_ns

        #stats = pstats.Stats(profiler).sort_stats('time')
        #stats.print_stats()

        mocked_upload_span.assert_called()

        print(f"took_ns: {took_ns}, calls: {calls}")
        self.assertTrue(took_ns / calls < 500 * 1e3) # less than 500 microseconds per trace
