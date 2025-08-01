import unittest
import logging
import time
from unittest.mock import patch, Mock
import pprint

import graphsignal
from graphsignal.uploader import Uploader
from graphsignal.spans import Span
from graphsignal.tracer import ProfilingTokenBucket
from test.model_utils import find_tag

logger = logging.getLogger('graphsignal')

class ProfilingTokenBucketTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_initialization(self):
        bucket = ProfilingTokenBucket(60)
        self.assertEqual(bucket.capacity, 60)
        self.assertEqual(bucket.tokens, 60)
        self.assertEqual(bucket.refill_rate_per_sec, 1.0)

    def test_should_profile_initial_call(self):
        bucket = ProfilingTokenBucket(10)
        # First call should be skipped
        self.assertFalse(bucket.should_profile())
        # Second call should be allowed
        self.assertTrue(bucket.should_profile())

    def test_should_profile_consumes_tokens(self):
        bucket = ProfilingTokenBucket(10)
        # First call should be skipped and not consume tokens
        self.assertFalse(bucket.should_profile())
        self.assertEqual(bucket.tokens, 10)  # No tokens consumed
        
        # Second call should be allowed and consume tokens
        self.assertTrue(bucket.should_profile())
        self.assertAlmostEqual(bucket.tokens, 9, delta=0.001)
        
        # Third call should also consume tokens
        self.assertTrue(bucket.should_profile())
        self.assertAlmostEqual(bucket.tokens, 8, delta=0.001)

    def test_should_profile_rate_limiting(self):
        bucket = ProfilingTokenBucket(1)
        # First call should be skipped
        self.assertFalse(bucket.should_profile())
        self.assertEqual(bucket.tokens, 1)  # No tokens consumed
        
        # Second call should consume the only token
        self.assertTrue(bucket.should_profile())
        self.assertLess(bucket.tokens, 0.1)
        
        # Third call should be rate limited
        self.assertFalse(bucket.should_profile())
        self.assertLess(bucket.tokens, 0.1)

    def test_token_refill(self):
        bucket = ProfilingTokenBucket(60)
        # Skip first request
        bucket.should_profile()  # This returns False, no tokens consumed
        
        # Consume all tokens (60 calls, but first was skipped)
        for _ in range(60):
            bucket.should_profile()
        self.assertLess(bucket.tokens, 0.1)
        
        bucket.last_refill_time = time.monotonic() - 1.1
        self.assertTrue(bucket.should_profile())
        self.assertGreater(bucket.tokens, 0)

    def test_token_bucket_capacity_limit(self):
        bucket = ProfilingTokenBucket(10)
        bucket.last_refill_time = time.monotonic() - 2.0
        self.assertEqual(bucket.tokens, 10)

    def test_different_rates(self):
        bucket30 = ProfilingTokenBucket(30)
        self.assertEqual(bucket30.refill_rate_per_sec, 0.5)
        
        bucket120 = ProfilingTokenBucket(120)
        self.assertEqual(bucket120.refill_rate_per_sec, 2.0)

    def test_first_request_skipped(self):
        bucket = ProfilingTokenBucket(10)
        
        # First request should be skipped
        self.assertFalse(bucket.should_profile())
        self.assertEqual(bucket.tokens, 10)  # No tokens consumed
        
        # Second request should be allowed
        self.assertTrue(bucket.should_profile())
        self.assertAlmostEqual(bucket.tokens, 9, delta=0.001)  # One token consumed
        
        # Third request should also be allowed
        self.assertTrue(bucket.should_profile())
        self.assertAlmostEqual(bucket.tokens, 8, delta=0.001)  # Another token consumed

class TracerTest(unittest.TestCase):
    def setUp(self):
        graphsignal.configure(
            api_key='k1',
            debug_mode=True)
        graphsignal._tracer.auto_export = False

    def tearDown(self):
        graphsignal.shutdown()

    @patch.object(Uploader, 'upload_metric')
    @patch.object(Uploader, 'upload_log_entry')
    def test_shutdown_upload(self, mocked_upload_log_entry, mocked_upload_metric):
        graphsignal.shutdown()
        graphsignal.configure(
            api_key='k1',
            debug_mode=True)
        graphsignal._tracer.metric_store().set_gauge(name='n1', tags={}, value=1, update_ts=1)
        graphsignal.shutdown()

        self.assertTrue(mocked_upload_metric.call_count > 0)

    @patch('graphsignal.tracer.uuid_sha1', return_value='123')
    def test_context_tag(self, mocked_uuid_sha1):
        tracer = graphsignal._tracer
        
        tracer.set_context_tag('k1', 'v1')
        self.assertEqual(tracer.get_context_tag('k1'), 'v1')

        tracer.set_context_tag('k2', 'v2', append_uuid=True)
        self.assertEqual(tracer.get_context_tag('k2'), 'v2-123')

        tracer.remove_context_tag('k1')
        self.assertEqual(tracer.get_context_tag('k1'), None)

        tracer.set_context_tag('k2', None)
        self.assertEqual(tracer.get_context_tag('k2'), None)

    def test_set_profiling_mode_with_token_bucket(self):
        tracer = graphsignal._tracer
        
        tracer._profiling_mode = None
        
        # First call will be skipped
        result = tracer.set_profiling_mode('python.cprofile')
        self.assertIsInstance(result, bool)
        self.assertFalse(result)  # First call skipped
        
        # Second call should succeed
        result = tracer.set_profiling_mode('python.cprofile')
        self.assertIsInstance(result, bool)
        self.assertTrue(result)  # Second call succeeds
        
        # Third call should fail (profiling mode already set)
        result = tracer.set_profiling_mode('python.cprofile')
        self.assertFalse(result)

    def test_set_profiling_mode_token_bucket_rate_limiting(self):
        tracer = graphsignal._tracer
        
        tracer._profiling_mode = None
        
        # Create bucket and consume all tokens
        bucket = tracer._profiling_token_buckets.get('python.cprofile')
        if bucket is None:
            tracer.set_profiling_mode('python.cprofile')  # This will create the bucket
            bucket = tracer._profiling_token_buckets['python.cprofile']
        
        bucket.tokens = 0
        
        result = tracer.set_profiling_mode('python.cprofile')
        self.assertFalse(result)

    def test_set_profiling_mode_already_set_not_expired(self):
        tracer = graphsignal._tracer
        tracer._profiling_mode = time.time()
        result = tracer.set_profiling_mode('python.cprofile')
        self.assertFalse(result)

    def test_set_profiling_mode_expired(self):
        tracer = graphsignal._tracer
        tracer._profiling_mode = time.time() - (tracer.PROFILING_MODE_TIMEOUT_SEC + 1)
        # First call will be skipped, second call should succeed
        result = tracer.set_profiling_mode('python.cprofile')
        self.assertFalse(result)  # First call skipped
        result = tracer.set_profiling_mode('python.cprofile')
        self.assertTrue(result)  # Second call succeeds

    def test_unset_profiling_mode(self):
        tracer = graphsignal._tracer
        # First call will be skipped, second call should succeed
        tracer.set_profiling_mode('python.cprofile')  # This returns False
        tracer.set_profiling_mode('python.cprofile')  # This returns True and sets profiling mode
        self.assertTrue(tracer.is_profiling_mode())
        tracer.unset_profiling_mode()
        self.assertFalse(tracer.is_profiling_mode())

    def test_profiling_token_buckets_initialization(self):
        tracer = graphsignal._tracer
        self.assertIsNotNone(tracer._profiling_token_buckets)
        self.assertEqual(len(tracer._profiling_token_buckets), 0)  # Initially empty

    def test_dynamic_token_bucket_creation(self):
        tracer = graphsignal._tracer
        
        # Initially no buckets
        self.assertEqual(len(tracer._profiling_token_buckets), 0)
        
        # Create bucket dynamically
        result = tracer.set_profiling_mode('python.cprofile')
        self.assertIn('python.cprofile', tracer._profiling_token_buckets)
        self.assertEqual(tracer._profiling_token_buckets['python.cprofile'].capacity, tracer.profiles_per_min)

    def test_multiple_profile_types(self):
        tracer = graphsignal._tracer
        
        # Create different profile types
        tracer.set_profiling_mode('python.cprofile')
        tracer.set_profiling_mode('pytorch.profile')
        
        self.assertIn('python.cprofile', tracer._profiling_token_buckets)
        self.assertIn('pytorch.profile', tracer._profiling_token_buckets)
        self.assertEqual(len(tracer._profiling_token_buckets), 2)

    def test_token_buckets_independent_operation(self):
        tracer = graphsignal._tracer
        tracer._profiling_mode = None
        
        # Create buckets
        tracer.set_profiling_mode('python.cprofile')
        tracer.set_profiling_mode('pytorch.profile')
        
        python_bucket = tracer._profiling_token_buckets['python.cprofile']
        pytorch_bucket = tracer._profiling_token_buckets['pytorch.profile']
        
        # Consume tokens from python bucket
        python_bucket.tokens = 0
        
        # pytorch bucket should still have tokens
        self.assertGreater(pytorch_bucket.tokens, 0)
        
        # python bucket should be rate limited
        result = tracer.set_profiling_mode('python.cprofile')
        self.assertFalse(result)
        
        # pytorch bucket should still work
        result = tracer.set_profiling_mode('pytorch.profile')
        self.assertIsInstance(result, bool)

    @patch.object(Uploader, 'upload_span')
    @patch.object(Uploader, 'upload_issue')
    def test_report_issue(self, mocked_upload_issue, mocked_upload_span):
        graphsignal.set_tag('k2', 'v2')
        graphsignal.set_context_tag('k3', 'v3')

        span = Span(operation='op1')
        span.set_tag('k5', 'v5')
        graphsignal.report_issue(name='issue1', severity=3, description='c1', span=span)

        issue = mocked_upload_issue.call_args[0][0]

        self.assertTrue(issue.issue_id is not None and issue.issue_id != '')
        self.assertEqual(issue.span_id, span.span_id)
        self.assertEqual(issue.name, 'issue1')
        self.assertEqual(find_tag(issue, 'operation.name'), 'op1')
        self.assertIsNotNone(find_tag(issue, 'host.name'))
        self.assertIsNotNone(find_tag(issue, 'process.pid'))
        self.assertEqual(find_tag(issue, 'k2'), 'v2')
        self.assertEqual(find_tag(issue, 'k3'), 'v3')
        self.assertEqual(find_tag(issue, 'k5'), 'v5')
        self.assertEqual(issue.severity, 3)
        self.assertEqual(issue.description, 'c1')
        self.assertTrue(issue.create_ts > 0)