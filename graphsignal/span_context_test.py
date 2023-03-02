import unittest
import logging
import sys
import time
from unittest.mock import patch, Mock
import asyncio
import random

import graphsignal
from graphsignal.span_context import start_span, stop_span, get_current_span

logger = logging.getLogger('graphsignal')

class SpanContextTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        graphsignal.configure(
            api_key='k1',
            deployment='d1',
            upload_on_shutdown=False,
            debug_mode=True)

    async def asyncTearDown(self):
        graphsignal.shutdown()

    async def test_start_span(self):
        start_span(name='s1', start_ns=1000)
        start_span(name='s2', start_ns=1001)
        self.assertEqual(get_current_span().name, 's2')
        start_span(name='s3', start_ns=1002)
        self.assertEqual(get_current_span().name, 's3')
        stop_span(end_ns=1003)
        self.assertEqual(get_current_span().name, 's2')
        span2 = stop_span(end_ns=1004, has_exception=True)
        self.assertEqual(get_current_span().start_ns, 1000)
        self.assertEqual(span2.has_exception, True)
        self.assertEqual(span2.end_ns, 1004)
        stop_span(end_ns=1005)
        self.assertIsNone(get_current_span())

    async def test_start_span_async(self):
        async def func1(ep):
            start_span(ep, start_ns=1001)
            await asyncio.sleep(random.random() * 0.2)
            self.assertEqual(get_current_span().name, ep)
            stop_span(end_ns=1002)

        start_span(name='r1', start_ns=1000)

        tasks = []
        for i in range(10):
            tasks.append(func1('s{}'.format(i)))

        await asyncio.gather(*tasks)

        stop_span(end_ns=1003)

        