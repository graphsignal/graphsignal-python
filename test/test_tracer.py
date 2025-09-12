import unittest
import logging
import time
from unittest.mock import patch, Mock
import pprint

import graphsignal
from graphsignal.uploader import Uploader
from graphsignal.spans import Span
from graphsignal.tracer import SamplingTokenBucket, DynamicCycle
from test.model_utils import find_tag

logger = logging.getLogger('graphsignal')

class SamplingTokenBucketTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_initialization(self):
        bucket = SamplingTokenBucket(60)
        self.assertEqual(bucket.capacity, 60)
        self.assertEqual(bucket.tokens, 60)
        self.assertEqual(bucket.refill_rate_per_sec, 1.0)

    def test_should_sample_initial_call(self):
        bucket = SamplingTokenBucket(10)
        # First call should be allowed
        self.assertTrue(bucket.should_sample())
        # Second call should also be allowed
        self.assertTrue(bucket.should_sample())

    def test_should_sample_consumes_tokens(self):
        bucket = SamplingTokenBucket(10)
        # First call should be allowed and consume tokens
        self.assertTrue(bucket.should_sample())
        self.assertAlmostEqual(bucket.tokens, 9, delta=0.001)
        
        # Second call should also consume tokens
        self.assertTrue(bucket.should_sample())
        self.assertAlmostEqual(bucket.tokens, 8, delta=0.001)
        
        # Third call should also consume tokens
        self.assertTrue(bucket.should_sample())
        self.assertAlmostEqual(bucket.tokens, 7, delta=0.001)

    def test_should_sample_rate_limiting(self):
        bucket = SamplingTokenBucket(1)
        # First call should consume the only token
        self.assertTrue(bucket.should_sample())
        self.assertLess(bucket.tokens, 0.1)
        
        # Second call should be rate limited
        self.assertFalse(bucket.should_sample())
        self.assertLess(bucket.tokens, 0.1)
        
        # Third call should also be rate limited
        self.assertFalse(bucket.should_sample())
        self.assertLess(bucket.tokens, 0.1)

    def test_token_refill(self):
        bucket = SamplingTokenBucket(60)
        
        # Consume all tokens (60 calls)
        for _ in range(60):
            bucket.should_sample()
        self.assertLess(bucket.tokens, 0.1)
        
        bucket.last_refill_time = time.monotonic() - 1.1
        self.assertTrue(bucket.should_sample())
        self.assertGreater(bucket.tokens, 0)

    def test_token_bucket_capacity_limit(self):
        bucket = SamplingTokenBucket(10)
        bucket.last_refill_time = time.monotonic() - 2.0
        self.assertEqual(bucket.tokens, 10)

    def test_different_rates(self):
        bucket30 = SamplingTokenBucket(30)
        self.assertEqual(bucket30.refill_rate_per_sec, 0.5)
        
        bucket120 = SamplingTokenBucket(120)
        self.assertEqual(bucket120.refill_rate_per_sec, 2.0)

class DynamicCycleTest(unittest.TestCase):
    def test_cycle_through_keys(self):
        cycle = DynamicCycle()
        
        # Add keys
        cycle.add('key1')
        cycle.add('key2')
        cycle.add('key2')
        cycle.add('key3')
        
        # Should cycle through keys in order
        self.assertEqual(cycle.next(), 'key1')
        self.assertEqual(cycle.next(), 'key2')
        self.assertEqual(cycle.next(), 'key3')
        self.assertEqual(cycle.next(), 'key1')
        self.assertEqual(cycle.next(), 'key2')

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
                
        # Second call should succeed
        result = tracer.set_profiling_mode('profile.cpython')
        self.assertIsInstance(result, bool)
        self.assertTrue(result)  # Second call succeeds
        
        # Third call should fail (profiling mode already set)
        result = tracer.set_profiling_mode('profile.cpython')
        self.assertFalse(result)

    def test_set_profiling_mode_token_bucket_rate_limiting(self):
        tracer = graphsignal._tracer
        
        tracer._profiling_mode = None
        
        # Create bucket and consume all tokens
        bucket = tracer._sampling_token_buckets.get('profile.cpython')
        if bucket is None:
            tracer.set_profiling_mode('profile.cpython')  # This will create the bucket
            bucket = tracer._sampling_token_buckets['profile.cpython']
        
        bucket.tokens = 0
        
        result = tracer.set_profiling_mode('profile.cpython')
        self.assertFalse(result)

    def test_set_profiling_mode_already_set_not_expired(self):
        tracer = graphsignal._tracer
        tracer._profiling_mode = time.time()
        result = tracer.set_profiling_mode('profile.cpython')
        self.assertFalse(result)

    def test_set_profiling_mode_expired(self):
        tracer = graphsignal._tracer
        tracer._profiling_mode = time.time() - (tracer.PROFILING_MODE_TIMEOUT_SEC + 1)
        result = tracer.set_profiling_mode('profile.cpython')
        self.assertTrue(result)  # Second call succeeds

    def test_unset_profiling_mode(self):
        tracer = graphsignal._tracer
        tracer.set_profiling_mode('profile.cpython')  # This returns True and sets profiling mode
        self.assertTrue(tracer.is_profiling_mode())
        tracer.unset_profiling_mode()
        self.assertFalse(tracer.is_profiling_mode())

    def test_sampling_token_buckets_initialization(self):
        tracer = graphsignal._tracer
        self.assertIsNotNone(tracer._sampling_token_buckets)
        self.assertEqual(len(tracer._sampling_token_buckets), 0)  # Initially empty

    def test_dynamic_token_bucket_creation(self):
        tracer = graphsignal._tracer
        
        # Initially no buckets
        self.assertEqual(len(tracer._sampling_token_buckets), 0)
        
        # Create bucket dynamically
        result = tracer.set_profiling_mode('profile.cpython')
        self.assertIn('profile.cpython', tracer._sampling_token_buckets)
        self.assertEqual(tracer._sampling_token_buckets['profile.cpython'].capacity, tracer.profiles_per_min)

    def test_multiple_profile_types(self):
        tracer = graphsignal._tracer
        
        # Create different profile types
        tracer.set_profiling_mode('profile.cpython')
        tracer.set_profiling_mode('profile.pytorch')
        
        self.assertIn('profile.cpython', tracer._sampling_token_buckets)
        self.assertIn('profile.pytorch', tracer._sampling_token_buckets)
        self.assertEqual(len(tracer._sampling_token_buckets), 2)

    def test_token_buckets_independent_operation(self):
        tracer = graphsignal._tracer
        tracer._profiling_mode = None
        
        # Create buckets
        tracer.set_profiling_mode('profile.cpython')
        tracer.set_profiling_mode('profile.pytorch')
        
        python_bucket = tracer._sampling_token_buckets['profile.cpython']
        pytorch_bucket = tracer._sampling_token_buckets['profile.pytorch']
        
        # Consume tokens from python bucket
        python_bucket.tokens = 0
        
        # pytorch bucket should still have tokens
        self.assertGreater(pytorch_bucket.tokens, 0)
        
        # python bucket should be rate limited
        result = tracer.set_profiling_mode('profile.cpython')
        self.assertFalse(result)
        
        # pytorch bucket should still work
        result = tracer.set_profiling_mode('profile.pytorch')
        self.assertIsInstance(result, bool)

    @patch.object(Uploader, 'upload_span')
    @patch.object(Uploader, 'upload_error')
    def test_report_error(self, mocked_upload_error, mocked_upload_span):
        graphsignal.set_tag('k2', 'v2')
        graphsignal.set_context_tag('k3', 'v3')

        span = Span(name='op1')
        span.set_tag('k5', 'v5')
        graphsignal.report_error(name='error1', level='warning', message='c1', tags=span.get_tags())

        error = mocked_upload_error.call_args[0][0]

        self.assertTrue(error.error_id is not None and error.error_id != '')
        self.assertEqual(error.name, 'error1')
        self.assertIsNotNone(find_tag(error, 'host.name'))
        self.assertIsNotNone(find_tag(error, 'process.pid'))
        self.assertEqual(find_tag(error, 'k2'), 'v2')
        self.assertEqual(find_tag(error, 'k3'), 'v3')
        self.assertEqual(find_tag(error, 'k5'), 'v5')
        self.assertEqual(error.level, 'warning')
        self.assertEqual(error.message, 'c1')
        self.assertTrue(error.create_ts > 0)

    @patch.object(Uploader, 'upload_error')
    def test_report_error_rate_limiting(self, mocked_upload_error):
        tracer = graphsignal._tracer
        
        tracer._error_counter = 0
        tracer._error_counter_reset_time = time.time()
        
        for i in range(tracer.MAX_ERRORS_PER_MINUTE):
            tracer.report_error(name=f'error_{i}')
        
        self.assertEqual(mocked_upload_error.call_count, tracer.MAX_ERRORS_PER_MINUTE)
        
        tracer.report_error(name='rate_limited_error')
        
        self.assertEqual(mocked_upload_error.call_count, tracer.MAX_ERRORS_PER_MINUTE)
        
        self.assertEqual(tracer._error_counter, tracer.MAX_ERRORS_PER_MINUTE)

    @patch.object(Uploader, 'upload_error')
    def test_report_error_invalid_level(self, mocked_upload_error):
        tracer = graphsignal._tracer
        
        # Test with invalid level
        tracer.report_error(name='test_error', level='invalid_level')
        
        # Should not upload error due to invalid level
        self.assertEqual(mocked_upload_error.call_count, 0)
        
        # Test with valid level
        tracer.report_error(name='test_error', level='warning')
        
        # Should upload error with valid level
        self.assertEqual(mocked_upload_error.call_count, 1)
        error = mocked_upload_error.call_args[0][0]
        self.assertEqual(error.level, 'warning')
        
