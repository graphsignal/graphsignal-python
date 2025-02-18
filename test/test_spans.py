import unittest
import logging
import sys
import time
from unittest.mock import patch, Mock

import graphsignal
from graphsignal.spans import Span, encode_payload
from graphsignal.recorders.process_recorder import ProcessRecorder
from graphsignal.uploader import Uploader


logger = logging.getLogger('graphsignal')


class SpansTest(unittest.TestCase):
    def setUp(self):
        if len(logger.handlers) == 0:
            logger.addHandler(logging.StreamHandler(sys.stdout))
        graphsignal.configure(
            api_key='k1',
            deployment='d1',
            tags={'k1': 'v1'},
            record_payloads=True,
            upload_on_shutdown=False,
            debug_mode=True)
        graphsignal._tracer.hostname = 'h1'

    def tearDown(self):
        graphsignal.shutdown()

    @patch.object(ProcessRecorder, 'on_span_stop')
    @patch.object(Uploader, 'upload_span')
    @patch.object(Uploader, 'upload_score')
    def test_start_stop(self, mocked_upload_score, mocked_upload_span, mocked_process_on_span_stop):
        graphsignal.set_tag('k2', 'v2')

        graphsignal.set_context_tag('k3', 'v3')
        graphsignal.set_context_tag('k4', 'v4')

        for i in range(10):
            span = Span(
                operation='op1',
                tags={'k4': 4.0})
            span.set_tag('k5', 'v5')
            span.set_usage('c3', 3)
            span.set_payload('input', [[1, 2],[3, 4]])
            span.set_profile('prof1', 'fmt1', 'content1')
            time.sleep(0.01)
            span.first_token()
            span.set_output_tokens(10)
            span.score(name='test-score', score=0.5, unit='u1', severity=3, comment='c1')
            time.sleep(0.01)
            span.stop()

        span = mocked_upload_span.call_args[0][0]

        self.assertTrue(span.start_us > 0)
        self.assertTrue(span.end_us > 0)
        self.assertTrue(span.latency_ns > 0)
        self.assertTrue(span.ttft_ns > 0)
        self.assertEqual(span.output_tokens, 10)
        self.assertEqual(span.root_span_id, span.span_id)
        self.assertEqual(find_tag(span, 'deployment'), 'd1')
        self.assertEqual(find_tag(span, 'operation'), 'op1')
        self.assertIsNotNone(find_tag(span, 'platform'))
        self.assertIsNotNone(find_tag(span, 'runtime'))
        self.assertIsNotNone(find_tag(span, 'hostname'))
        self.assertIsNotNone(find_tag(span, 'process_id'))
        self.assertEqual(find_tag(span, 'k1'), 'v1')
        self.assertEqual(find_tag(span, 'k2'), 'v2')
        self.assertEqual(find_tag(span, 'k3'), 'v3')
        self.assertEqual(find_tag(span, 'k4'), '4.0')
        self.assertEqual(find_tag(span, 'k5'), 'v5')
        self.assertEqual(find_usage(span, 'c3'), 3)
        self.assertEqual(find_payload(span, 'input').content_type, 'application/json')
        self.assertEqual(find_payload(span, 'input').content_base64, 'W1sxLCAyXSwgWzMsIDRdXQ==')
        self.assertEqual(find_profile(span, 'prof1').format, 'fmt1')
        self.assertEqual(find_profile(span, 'prof1').content, 'content1')


        store = graphsignal._tracer.metric_store()
        metric_tags =  graphsignal._tracer.tags.copy()
        metric_tags['operation'] = 'op1'
        metric_tags['k3'] = 'v3'  
        metric_tags['k4'] = 4.0
        metric_tags['k5'] = 'v5'
        key = store.metric_key('performance', 'latency', metric_tags)
        self.assertTrue(len(store._metrics[key].histogram) > 0)
        key = store.metric_key('performance', 'first_token', metric_tags)
        self.assertTrue(len(store._metrics[key].histogram) > 0)
        key = store.metric_key('performance', 'call_count', metric_tags)
        self.assertEqual(store._metrics[key].counter, 10)
        key = store.metric_key('performance', 'output_tps', metric_tags)
        self.assertEqual(store._metrics[key].count, 100)
        self.assertTrue(store._metrics[key].interval > 0)

        usage_tags = metric_tags.copy()
        key = store.metric_key('usage', 'c3', usage_tags)
        self.assertEqual(store._metrics[key].counter, 30)

        score = mocked_upload_score.call_args[0][0]

        self.assertTrue(score.score_id is not None and score.score_id != '')
        self.assertEqual(score.span_id, span.span_id)
        self.assertEqual(score.name, 'test-score')
        self.assertEqual(find_tag(score, 'deployment'), 'd1')
        self.assertEqual(find_tag(score, 'operation'), 'op1')
        self.assertIsNotNone(find_tag(score, 'platform'))
        self.assertIsNotNone(find_tag(score, 'runtime'))
        self.assertIsNotNone(find_tag(score, 'hostname'))
        self.assertIsNotNone(find_tag(score, 'process_id'))
        self.assertEqual(find_tag(score, 'k1'), 'v1')
        self.assertEqual(find_tag(score, 'k2'), 'v2')
        self.assertEqual(find_tag(score, 'k3'), 'v3')
        self.assertEqual(find_tag(score, 'k4'), '4.0')
        self.assertEqual(find_tag(score, 'k5'), 'v5')        
        self.assertEqual(score.score, 0.5)
        self.assertEqual(score.unit, 'u1')
        self.assertEqual(score.severity, 3)
        self.assertEqual(score.comment, 'c1')
        self.assertTrue(score.create_ts > 0)

    @patch.object(ProcessRecorder, 'on_span_start')
    @patch.object(Uploader, 'upload_span')
    def test_start_stop_nested(self, mocked_upload_span, mocked_process_on_span_start):
        with Span(operation='op1') as span:
            with span.trace(operation='op2'):
                pass

        t1 = mocked_upload_span.call_args_list[1][0][0]
        t2 = mocked_upload_span.call_args_list[0][0][0]

        self.assertTrue(t1.start_us > 0)
        self.assertTrue(t1.end_us > 0)
        self.assertEqual(t1.root_span_id, t1.span_id)
        self.assertEqual(t2.parent_span_id, t1.span_id)
        self.assertEqual(t2.root_span_id, t1.span_id)

        self.assertTrue(t2.start_us > 0)
        self.assertTrue(t2.end_us > 0)


    @patch.object(ProcessRecorder, 'on_span_start')
    @patch.object(Uploader, 'upload_span')
    def test_start_exception(self, mocked_upload_span, mocked_process_on_span_start):
        mocked_process_on_span_start.side_effect = Exception('ex1')

        store = graphsignal._tracer.log_store()
        store.clear()

        span = Span(operation='op1')
        span.stop()
        span = mocked_upload_span.call_args[0][0]

        self.assertTrue(span.start_us > 0)
        self.assertTrue(span.end_us > 0)
        self.assertEqual(span.root_span_id, span.span_id)
        self.assertEqual(find_log_entry(store, 'ex1').tags['deployment'], 'd1')
        self.assertIsNotNone(find_log_entry(store, 'ex1').message)
        self.assertIsNotNone(find_log_entry(store, 'ex1').exception)

    @patch.object(ProcessRecorder, 'on_span_stop')
    @patch.object(Uploader, 'upload_span')
    def test_stop_exception(self, mocked_upload_span, mocked_process_on_span_stop):
        mocked_process_on_span_stop.side_effect = Exception('ex1')

        store = graphsignal._tracer.log_store()
        store.clear()

        span = Span(operation='op1')
        span.stop()

        span = mocked_upload_span.call_args[0][0]

        self.assertTrue(span.start_us > 0)
        self.assertTrue(span.end_us > 0)
        self.assertEqual(find_log_entry(store, 'ex1').tags['deployment'], 'd1')
        self.assertIsNotNone(find_log_entry(store, 'ex1').message)
        self.assertIsNotNone(find_log_entry(store, 'ex1').exception)

    @patch.object(Uploader, 'upload_span')
    def test_operation_exception(self, mocked_upload_span):
        for _ in range(2):
            try:
                with Span(operation='op1'):
                    raise Exception('ex1')
            except Exception as ex:
                if str(ex) != 'ex1':
                    raise ex

        self.assertEqual(mocked_upload_span.call_count, 2)
        span = mocked_upload_span.call_args[0][0]

        self.assertTrue(span.span_id != '')
        self.assertTrue(span.start_us > 0)
        self.assertTrue(span.end_us > 0)
        self.assertEqual(span.root_span_id, span.span_id)
        self.assertEqual(span.exceptions[0].exc_type, 'Exception')
        self.assertEqual(span.exceptions[0].message, 'ex1')
        self.assertNotEqual(span.exceptions[0].stack_trace, '')

        store = graphsignal._tracer.metric_store()
        metric_tags = graphsignal._tracer.tags.copy()
        metric_tags['operation'] = 'op1'
        key = store.metric_key('performance', 'exception_count', metric_tags)
        self.assertEqual(store._metrics[key].counter, 2)

    @patch.object(Uploader, 'upload_span')
    def test_add_exception(self, mocked_upload_span):
        span = Span(operation='op1')
        try:
            raise Exception('ex2')
        except Exception as ex:
            span.add_exception(ex)
        span.stop()

        span = mocked_upload_span.call_args[0][0]

        self.assertTrue(span.start_us > 0)
        self.assertTrue(span.end_us > 0)
        self.assertEqual(span.root_span_id, span.span_id)
        self.assertEqual(span.exceptions[0].exc_type, 'Exception')
        self.assertEqual(span.exceptions[0].message, 'ex2')
        self.assertNotEqual(span.exceptions[0].stack_trace, '')

        store = graphsignal._tracer.metric_store()
        metric_tags = graphsignal._tracer.tags.copy()
        metric_tags['operation'] = 'op1'
        key = store.metric_key('performance', 'exception_count', metric_tags)
        self.assertEqual(store._metrics[key].counter, 1)

    @patch.object(Uploader, 'upload_span')
    def test_add_exception_true(self, mocked_upload_span):
        span = Span(operation='op1')
        try:
            raise Exception('ex2')
        except Exception as ex:
            span.add_exception(exc_info=True)
        span.stop()

        span = mocked_upload_span.call_args[0][0]

        self.assertTrue(span.start_us > 0)
        self.assertTrue(span.end_us > 0)
        self.assertEqual(span.root_span_id, span.span_id)
        self.assertEqual(span.exceptions[0].exc_type, 'Exception')
        self.assertEqual(span.exceptions[0].message, 'ex2')
        self.assertNotEqual(span.exceptions[0].stack_trace, '')

        store = graphsignal._tracer.metric_store()
        metric_tags = graphsignal._tracer.tags.copy()
        metric_tags['operation'] = 'op1'
        key = store.metric_key('performance', 'exception_count', metric_tags)
        self.assertEqual(store._metrics[key].counter, 1)

    @patch.object(Uploader, 'upload_span')
    def test_set_payload(self, mocked_upload_span):
        with Span(operation='op1') as span:
            span.set_payload('d1', None)

        span = mocked_upload_span.call_args[0][0]

        self.assertTrue(span.start_us > 0)
        self.assertTrue(span.end_us > 0)
        self.assertEqual(span.root_span_id, span.span_id)

    @patch.object(Uploader, 'upload_span')
    def test_subspans(self, mocked_upload_span):
        with Span(operation='op1') as span1:
            with span1.trace(operation='op2') as span2:
                with span2.trace(operation='op3'):
                    pass
            with span1.trace(operation='ep4'):
                pass

        t3 = mocked_upload_span.call_args_list[0][0][0]
        t2 = mocked_upload_span.call_args_list[1][0][0]
        t4 = mocked_upload_span.call_args_list[2][0][0]
        t1 = mocked_upload_span.call_args_list[3][0][0]

        self.assertEqual(t1.parent_span_id, None)
        self.assertEqual(t1.root_span_id, t1.span_id)

        self.assertEqual(t2.parent_span_id, t1.span_id)
        self.assertEqual(t2.root_span_id, t1.span_id)

        self.assertEqual(t3.parent_span_id, t2.span_id)
        self.assertEqual(t3.root_span_id, t1.span_id)

        self.assertEqual(t4.parent_span_id, t1.span_id)
        self.assertEqual(t4.root_span_id, t1.span_id)

    @unittest.skip('for now')
    @patch.object(Uploader, 'upload_span')
    def test_overhead(self, mocked_upload_span):
        #import cProfile, pstats
        #profiler = cProfile.Profile()
        #profiler.enable()

        graphsignal._tracer.debug_mode = False

        calls = 10000
        start_ns = time.perf_counter_ns()
        for _ in range(calls):
            with Span(operation='op1') as span:
                span.set_tag('k1', 'v1')
                span.set_payload('test', 'test string data')
        took_ns = time.perf_counter_ns() - start_ns

        #stats = pstats.Stats(profiler).sort_stats('time')
        #stats.print_stats()

        self.assertTrue(took_ns / calls < 200 * 1e3) # less than 200 microseconds per trace

    def test_encode_payload(self):
        content_type, content_bytes = encode_payload(['text\n', 2.0, float('nan')])
        self.assertEqual(content_type, 'application/json')
        self.assertEqual(content_bytes, b'["text\\n", 2.0, NaN]')


def find_tag(model, key):
    for tag in model.tags:
        if tag.key == key:
            return tag.value
    return None


def find_usage(model, name):
    for counter in model.usage:
        if counter.name == name:
            return counter.value
    return None


def find_payload(model, name):
    for payload in model.payloads:
        if payload.name == name:
            return payload
    return None

def find_profile(model, name):
    for profile in model.profiles:
        if profile.name == name:
            return profile
    return None

def find_log_entry(store, text):
    for entry in store._logs:
        if entry.message and text in entry.message:
            return entry
        if entry.exception and text in entry.exception:
            return entry
