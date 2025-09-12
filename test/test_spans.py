import unittest
import logging
import sys
import time
from unittest.mock import patch, Mock

import graphsignal
from graphsignal.spans import Span, SpanContext
from graphsignal.recorders.process_recorder import ProcessRecorder
from graphsignal.uploader import Uploader
from test.model_utils import find_tag, find_attribute, find_counter, find_profile, find_log_entry


logger = logging.getLogger('graphsignal')


class SpansTest(unittest.TestCase):
    def setUp(self):
        if len(logger.handlers) == 0:
            logger.addHandler(logging.StreamHandler(sys.stdout))
        graphsignal.configure(
            api_key='k1',
            tags={'deployment': 'd1', 'k1': 'v1'},
            debug_mode=True)
        graphsignal._tracer.hostname = 'h1'
        graphsignal._tracer.auto_export = False

    def tearDown(self):
        graphsignal.shutdown()

    @patch.object(ProcessRecorder, 'on_span_stop')
    @patch.object(Uploader, 'upload_span')
    @patch.object(Uploader, 'upload_profile')
    def test_start_stop(self, mocked_upload_profile, mocked_upload_span, mocked_process_on_span_stop):
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
            span.set_profile('prof1', 'fmt1', 'content1')
            span.set_sampled(True)
            time.sleep(0.01)
            span.set_counter('output_tokens', 10)
            time.sleep(0.01)
            span.stop()

        span = mocked_upload_span.call_args[0][0]

        self.assertTrue(span.start_ns > 0)
        self.assertTrue(span.end_ns > 0)
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
        
        # Check that profile was uploaded separately
        mocked_upload_profile.assert_called()
        
        # Find the profile with name 'prof1' among all uploaded profiles
        prof1_profile = None
        for call in mocked_upload_profile.call_args_list:
            profile = call[0][0]
            if profile.name == 'prof1':
                prof1_profile = profile
                break
        
        self.assertIsNotNone(prof1_profile, "Profile 'prof1' not found in uploaded profiles")
        self.assertEqual(prof1_profile.name, 'prof1')
        self.assertEqual(prof1_profile.format, 'fmt1')
        self.assertEqual(prof1_profile.content, 'content1')
        self.assertEqual(len(prof1_profile.linked_span_ids), 1)
        self.assertTrue(prof1_profile.start_ns > 0)
        self.assertTrue(prof1_profile.end_ns > 0)

        store = graphsignal._tracer.metric_store()
        metric_tags =  graphsignal._tracer.tags.copy()
        metric_tags['span.name'] = 'op1'
        metric_tags['k3'] = 'v3'  
        metric_tags['k4'] = 4.0
        metric_tags['k5'] = 'v5'
        key = store.metric_key('span.call.count', metric_tags)
        self.assertEqual(store._metrics[key].total, 10)

        key = store.metric_key('c3', metric_tags)
        self.assertEqual(store._metrics[key].total, 30)

    @patch.object(ProcessRecorder, 'on_span_start')
    @patch.object(Uploader, 'upload_span')
    def test_start_stop_contextvars(self, mocked_upload_span, mocked_process_on_span_start):
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
        self.assertEqual(t1.parent_span_id, None)

        self.assertEqual(t2.trace_id, t1.trace_id)
        self.assertEqual(t2.parent_span_id, t1.span_id)


    @patch.object(ProcessRecorder, 'on_span_start')
    @patch.object(Uploader, 'upload_span')
    def test_start_exception(self, mocked_upload_span, mocked_process_on_span_start):
        mocked_process_on_span_start.side_effect = Exception('ex1')

        store = graphsignal._tracer.log_store()
        store.clear()

        span = Span(name='op1')
        span.set_sampled(True)
        span.stop()
        span = mocked_upload_span.call_args[0][0]

        self.assertTrue(span.start_ns > 0)
        self.assertTrue(span.end_ns > 0)
        self.assertEqual(find_log_entry(store, 'ex1').tags['deployment'], 'd1')
        self.assertIsNotNone(find_log_entry(store, 'ex1').message)
        self.assertIsNotNone(find_log_entry(store, 'ex1').exception)

    @patch.object(ProcessRecorder, 'on_span_stop')
    @patch.object(Uploader, 'upload_span')
    def test_stop_exception(self, mocked_upload_span, mocked_process_on_span_stop):
        mocked_process_on_span_stop.side_effect = Exception('ex1')

        store = graphsignal._tracer.log_store()
        store.clear()

        span = Span(name='op1')
        span.set_sampled(True)
        span.stop()

        span = mocked_upload_span.call_args[0][0]

        self.assertTrue(span.start_ns > 0)
        self.assertTrue(span.end_ns > 0)
        self.assertEqual(find_log_entry(store, 'ex1').tags['deployment'], 'd1')
        self.assertIsNotNone(find_log_entry(store, 'ex1').message)
        self.assertIsNotNone(find_log_entry(store, 'ex1').exception)

    @patch.object(Uploader, 'upload_error')
    @patch.object(Uploader, 'upload_span')
    def test_span_exception(self, mocked_upload_span, mocked_upload_error):
        for _ in range(2):
            try:
                with Span(name='op1') as span:
                    span.set_sampled(True)
                    raise Exception('ex1')
            except Exception as ex:
                if str(ex) != 'ex1':
                    raise ex

        self.assertEqual(mocked_upload_span.call_count, 2)
        self.assertEqual(mocked_upload_error.call_count, 2)

        # Check that errors were reported via report_error
        span = mocked_upload_span.call_args[0][0]
        error = mocked_upload_error.call_args[0][0]
        self.assertEqual(error.name, 'span.error')
        self.assertEqual(error.level, 'error')
        self.assertEqual(error.linked_span_ids, [span.span_id])
        self.assertIn('Exception: ex1', error.message)

        store = graphsignal._tracer.metric_store()
        metric_tags = graphsignal._tracer.tags.copy()
        metric_tags['span.name'] = 'op1'
        key = store.metric_key('span.error.count', metric_tags)
        self.assertEqual(store._metrics[key].total, 2)

    @patch.object(Uploader, 'upload_span')
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
        self.assertEqual(t1.parent_span_id, None)

        self.assertEqual(t2.trace_id, t1.trace_id)
        self.assertEqual(t2.parent_span_id, t1.span_id)

        self.assertEqual(t4.trace_id, t1.trace_id)
        self.assertEqual(t3.parent_span_id, t2.span_id)

        self.assertEqual(t4.trace_id, t1.trace_id)
        self.assertEqual(t4.parent_span_id, t1.span_id)

    @unittest.skip('for now')
    @patch.object(Uploader, 'upload_span')
    def test_overhead(self, mocked_upload_span):
        #import cProfile, pstats
        #profiler = cProfile.Profile()
        #profiler.enable()

        graphsignal._tracer.debug_mode = False
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

        #print(f"took_ns: {took_ns}, calls: {calls}")
        self.assertTrue(took_ns / calls < 200 * 1e3) # less than 200 microseconds per trace
