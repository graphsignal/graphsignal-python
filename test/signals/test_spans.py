import unittest
import logging
import sys
import time
from unittest.mock import patch, Mock

import graphsignal
from graphsignal.signals.spans import Span, SpanContext
from graphsignal.core.signal_uploader import SignalUploader
from test.test_utils import find_tag, find_attribute, find_counter, find_last_datapoint


logger = logging.getLogger('graphsignal')


class SpansTest(unittest.TestCase):
    def setUp(self):
        if len(logger.handlers) == 0:
            logger.addHandler(logging.StreamHandler(sys.stdout))
        graphsignal.configure(
            api_key='k1',
            tags={'deployment': 'd1', 'k1': 'v1'},
            debug_mode=True)
        graphsignal._ticker.hostname = 'h1'
        graphsignal._ticker.auto_tick = False

    def tearDown(self):
        graphsignal.shutdown()

    @patch.object(SignalUploader, 'upload_span')
    def test_start_stop(self, mocked_upload_span):
        graphsignal.set_tag('k2', 'v2')

        graphsignal.set_context_tag('k3', 'v3')
        graphsignal.set_context_tag('k4', 'v4')

        for i in range(10):
            span = Span(
                name='op1',
                tags={'k4': 4.0})
            span.set_tag('k5', 'v5')
            span.set_attribute('p3', 'v3')
            span.set_counter('c3', 3)
            span.inc_counter_metric('c3', 3)
            span.set_sampled(True)
            time.sleep(0.01)
            span.set_counter('output_tokens', 10)
            time.sleep(0.01)
            span.stop()

        span = mocked_upload_span.call_args[0][0]

        self.assertTrue(span.start_ts > 0)
        self.assertTrue(span.end_ts > 0)
        self.assertEqual(span.name, 'op1')
        self.assertEqual(find_tag(span, 'deployment'), 'd1')
        self.assertIsNotNone(find_tag(span, 'host.name'))
        self.assertIsNotNone(find_tag(span, 'process.pid'))
        self.assertIsNotNone(find_tag(span, 'platform.name'))
        self.assertIsNotNone(find_tag(span, 'platform.version'))
        self.assertIsNotNone(find_tag(span, 'runtime.name'))
        self.assertIsNotNone(find_tag(span, 'runtime.version'))
        self.assertEqual(find_attribute(span, 'p3'), 'v3')
        self.assertEqual(find_tag(span, 'k1'), 'v1')
        self.assertEqual(find_tag(span, 'k2'), 'v2')
        self.assertEqual(find_tag(span, 'k3'), 'v3')
        self.assertEqual(find_tag(span, 'k4'), '4.0')
        self.assertEqual(find_tag(span, 'k5'), 'v5')
        self.assertTrue(find_counter(span, 'span.duration') > 0)
        self.assertEqual(find_counter(span, 'c3'), 3)
        
        store = graphsignal._ticker.metric_store()
        metric_tags =  graphsignal._ticker.tags.copy()
        metric_tags['span.name'] = 'op1'
        key = store.metric_key('span.call.count', metric_tags)
        self.assertEqual(find_last_datapoint(store, key).total, 10)

        key = store.metric_key('c3', metric_tags)
        self.assertEqual(find_last_datapoint(store, key).total, 30)

    @patch.object(SignalUploader, 'upload_span')
    def test_start_stop_contextvars(self, mocked_upload_span):
        with Span(name='op1') as span1:
            span1.set_sampled(True)
            ctx1 = span1.get_span_context()
            SpanContext.push_contextvars(ctx1)

            ctx1_copy = SpanContext.pop_contextvars()
            with Span(name='op2', parent_context=ctx1_copy) as span2:
                ctx2 = span2.get_span_context()
                self.assertEqual(ctx1.trace_id, ctx2.trace_id)

        t1 = mocked_upload_span.call_args_list[1][0][0]
        t2 = mocked_upload_span.call_args_list[0][0][0]

        self.assertIsNotNone(t1.trace_id)
        # Protobuf returns empty string for unset optional fields
        self.assertEqual(t1.parent_span_id, '')

        self.assertEqual(t2.trace_id, t1.trace_id)
        self.assertEqual(t2.parent_span_id, t1.span_id)

    @patch.object(SignalUploader, 'upload_span')
    def test_span_exception(self, mocked_upload_span):
        for _ in range(2):
            try:
                with Span(name='op1') as span:
                    span.set_sampled(True)
                    raise Exception('ex1')
            except Exception as ex:
                if str(ex) != 'ex1':
                    raise ex

        self.assertEqual(mocked_upload_span.call_count, 2)

        # Check that exceptions were saved as SpanEvent in both spans
        for call in mocked_upload_span.call_args_list:
            span = call[0][0]

            self.assertEqual(find_tag(span, 'span.status'), 'error')

            self.assertEqual(len(span.events), 1)

            event = span.events[0]
            self.assertTrue(event.event_ts > 0)
            self.assertEqual(event.name, 'exception')
            # Check event attributes
            event_attrs = {attr.name: attr.value for attr in event.attributes}
            self.assertEqual(event_attrs['exception.type'], 'Exception')
            self.assertEqual(event_attrs['exception.message'], 'ex1')
            self.assertIsNotNone(event_attrs.get('exception.stacktrace'))

        store = graphsignal._ticker.metric_store()
        metric_tags = graphsignal._ticker.tags.copy()
        metric_tags['span.name'] = 'op1'
        key = store.metric_key('span.error.count', metric_tags)
        self.assertEqual(find_last_datapoint(store, key).total, 2)

    @patch.object(SignalUploader, 'upload_span')
    def test_subspans(self, mocked_upload_span):
        with Span(name='op1') as span1:
            span1.set_sampled(True)
            with span1.trace(span_name='op2') as span2:
                with span2.trace(span_name='op3'):
                    pass
            with span1.trace(span_name='ep4'):
                pass

        t3 = mocked_upload_span.call_args_list[0][0][0]
        t2 = mocked_upload_span.call_args_list[1][0][0]
        t4 = mocked_upload_span.call_args_list[2][0][0]
        t1 = mocked_upload_span.call_args_list[3][0][0]

        self.assertIsNotNone(t1.trace_id)
        self.assertEqual(t1.parent_span_id, '')

        self.assertEqual(t2.trace_id, t1.trace_id)
        self.assertEqual(t2.parent_span_id, t1.span_id)

        self.assertEqual(t4.trace_id, t1.trace_id)
        self.assertEqual(t3.parent_span_id, t2.span_id)

        self.assertEqual(t4.trace_id, t1.trace_id)
        self.assertEqual(t4.parent_span_id, t1.span_id)

    @unittest.skip('for now')
    @patch.object(SignalUploader, 'upload_span')
    def test_overhead(self, mocked_upload_span):
        #import cProfile, pstats
        #profiler = cProfile.Profile()
        #profiler.enable()

        graphsignal._ticker.debug_mode = False
        logger.setLevel(logging.ERROR)

        calls = 1000
        start_ns = time.perf_counter_ns()
        for _ in range(calls):
            with Span(name='op1') as span:
                span.set_sampled(True)
                span.set_tag('k1', 'v1')
        took_ns = time.perf_counter_ns() - start_ns

        #stats = pstats.Stats(profiler).sort_stats('time')
        #stats.print_stats()

        mocked_upload_span.assert_called()

        print(f"took_ns: {took_ns}, calls: {calls}")
        self.assertTrue(took_ns / calls < 200 * 1e3) # less than 200 microseconds per trace
