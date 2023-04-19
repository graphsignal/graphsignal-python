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
        s1 = Span(name='s1', start_ns=1000)
        s2 = Span(name='s2', start_ns=1001)
        self.assertEqual(get_current_span().name, 's2')
        self.assertEqual(get_current_span().total_count(), 2)
        s3 = Span(name='s3', start_ns=1002)
        self.assertEqual(get_current_span().name, 's3')
        self.assertEqual(get_current_span().total_count(), 3)
        s3.stop(end_ns=1003)
        self.assertEqual(get_current_span().name, 's2')
        s2.stop(end_ns=1004, trace_id='t1')
        self.assertEqual(get_current_span().start_ns, 1000)
        self.assertEqual(s2.trace_id, 't1')
        self.assertEqual(s2.end_ns, 1004)
        s1.stop(end_ns=1005)
        self.assertIsNone(get_current_span())

    async def test_span_async(self):
        async def func1(ep):
            s2 = Span(ep, start_ns=1001)
            await asyncio.sleep(random.random() * 0.2)
            self.assertEqual(get_current_span().name, ep)
            s2.stop(end_ns=1002)

        s1 = Span(name='r1', start_ns=1000)

        tasks = []
        for i in range(10):
            tasks.append(func1('s{i}'))

        await asyncio.gather(*tasks)

        self.assertEqual(get_current_span().total_count(), 11)
        s1.stop(end_ns=1003)

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
