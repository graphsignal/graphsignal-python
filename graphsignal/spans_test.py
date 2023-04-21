import unittest
import logging
import sys
import time
from unittest.mock import patch, Mock
import asyncio
import random

import graphsignal
from graphsignal.spans import get_current_span, Span

logger = logging.getLogger('graphsignal')

class SpanTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        graphsignal.configure(
            api_key='k1',
            deployment='d1',
            upload_on_shutdown=False,
            debug_mode=True)

    async def asyncTearDown(self):
        graphsignal.shutdown()

    async def test_span_nested(self):
        s1 = Span('s1')
        s1.set_trace_id('t1')
        s1.set_sampling(True)
        self.assertEqual(get_current_span().parent_span, None)
        self.assertEqual(get_current_span().root_span.trace_id, 't1')
        self.assertEqual(get_current_span().root_span.total_count, 1)
        self.assertTrue(get_current_span().is_root_sampling())
        s2 = Span('s2')
        s2.set_trace_id('t2')
        self.assertEqual(get_current_span().operation, 's2')
        self.assertEqual(get_current_span().trace_id, 't2')
        self.assertEqual(get_current_span().parent_span.trace_id, 't1')
        self.assertEqual(get_current_span().root_span.trace_id, 't1')
        self.assertEqual(get_current_span().root_span.total_count, 2)
        self.assertTrue(get_current_span().is_root_sampling())
        s3 = Span('s3')
        s3.set_trace_id('t3')
        self.assertEqual(get_current_span().operation, 's3')
        self.assertEqual(get_current_span().trace_id, 't3')
        self.assertEqual(get_current_span().parent_span.trace_id, 't2')
        self.assertEqual(get_current_span().root_span.trace_id, 't1')
        self.assertEqual(get_current_span().root_span.total_count, 3)
        self.assertTrue(get_current_span().is_root_sampling())
        s3.stop()
        self.assertEqual(get_current_span().operation, 's2')
        s2.set_trace_id('t1')
        s2.stop()
        self.assertEqual(s2.trace_id, 't1')
        s1.stop()
        self.assertIsNone(get_current_span())

    async def test_span_async(self):
        async def func1(ep):
            s2 = Span(ep)
            await asyncio.sleep(random.random() * 0.2)
            self.assertEqual(get_current_span().operation, ep)
            s2.stop()

        s1 = Span('r1')

        tasks = []
        for i in range(10):
            tasks.append(func1('s{i}'))

        await asyncio.gather(*tasks)

        self.assertEqual(get_current_span().root_span.total_count, 11)
        s1.stop()

    async def test_span_limit(self):
        for i in range(Span.MAX_NESTED_SPANS + 10):
            s = Span(f's{i}')
            if i < Span.MAX_NESTED_SPANS - 1:
                self.assertTrue(s.can_add_child())
                self.assertTrue(s.in_context)
            elif i == Span.MAX_NESTED_SPANS - 1:
                self.assertFalse(s.can_add_child())
                self.assertTrue(s.in_context)
            else:
                self.assertFalse(s.can_add_child())
                self.assertFalse(s.in_context)
