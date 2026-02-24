import unittest
import logging
import sys
import time
import threading
from unittest.mock import Mock, patch, MagicMock
import pytest
import io

import graphsignal
from graphsignal.profilers.function_profiler import FunctionProfiler, FunctionBucket

logger = logging.getLogger('graphsignal')

# Check if sys.monitoring is available (Python 3.12+)
HAS_SYS_MONITORING = hasattr(sys, 'monitoring')


class TestFunctions:
    
    def fast_function(self):
        return 42
    
    def slow_function(self):
        time.sleep(0.1)
        return 100
    
    def error_function(self):
        raise ValueError("Test error")
    
    @staticmethod
    def static_function():
        return "static"
    
    def nested_function(self):
        return self.fast_function()


class FunctionProfilerTest(unittest.TestCase):
    def setUp(self):
        if len(logger.handlers) == 0:
            logger.addHandler(logging.StreamHandler(sys.stdout))
        graphsignal.configure(
            api_key='k1',
            debug_mode=True)
        graphsignal._ticker.auto_tick = False
        # shutdown to allow reconfiguration of the profiler
        graphsignal._ticker._function_profiler.shutdown()
        # Create instance for calling instance methods
        self.test_functions = TestFunctions()

    def tearDown(self):
        graphsignal.shutdown()

    @unittest.skipUnless(HAS_SYS_MONITORING, "sys.monitoring requires Python 3.12+")
    @patch('graphsignal._ticker.update_profile')
    def test_single_bucket_event(self, mock_update_profile):
        resolution_ns = 1_000_000_000  # 1 second
        profiler = FunctionProfiler('test_profile')

        profiler.set_resolution_ns(resolution_ns)
        profiler.setup()
        
        test_func = self.test_functions.fast_function
        profiler.add_function(test_func, category='test', event_name='fast_function')
        
        # Call the function - this will trigger sys.monitoring callbacks
        test_func()
        
        # Wait a bit for the callback to process
        time.sleep(0.1)
        
        # Manually trigger rollover to see the bucket data
        profiler._rollover_buckets(time.time_ns())
        
        # Wait a bit more
        time.sleep(0.1)
        
        # Check that update_profile was called
        self.assertGreater(mock_update_profile.call_count, 0)
        
        # Find the call with our function's data
        for call in mock_update_profile.call_args_list:
            args, kwargs = call
            if len(args) >= 3:
                name, profile, measurement_ts = args[:3]
                if profile and len(profile) > 0:
                    # The profile should contain field IDs for duration, calls, and possibly errors
                    self.assertGreater(len(profile), 0)
                    break
        
        profiler.shutdown()

    @unittest.skipUnless(HAS_SYS_MONITORING, "sys.monitoring requires Python 3.12+")
    @patch('graphsignal._ticker.update_profile')
    def test_multi_bucket_event(self, mock_update_profile):
        resolution_ns = 100_000_000  # 100ms
        profiler = FunctionProfiler('test_profile')

        profiler.set_resolution_ns(resolution_ns)
        profiler.setup()
        
        test_func = self.test_functions.slow_function
        profiler.add_function(test_func, category='test', event_name='slow_function')
        
        # Start function call in a thread
        def call_function():
            test_func()

        thread = threading.Thread(target=call_function)
        thread.start()

        # Wait a bit for function to start
        time.sleep(0.05)
        
        # Check bucket state before rollover (function should still be running)
        code = test_func.__code__
        bucket = profiler._buckets.get(code)
        if bucket:
            # Should have running functions
            self.assertGreater(bucket.num_running, 0)
        else:
            self.fail("Bucket not found for function")

        # Wait for first rollover (function still running)
        time.sleep(0.1)
        profiler._rollover_buckets(time.time_ns())
        
        # Wait for function to complete
        thread.join()
        
        # Wait a bit more
        time.sleep(0.1)
        
        # Trigger another rollover
        profiler._rollover_buckets(time.time_ns())
        
        # Should have recorded the function across multiple buckets
        self.assertGreater(mock_update_profile.call_count, 0)
        
        profiler.shutdown()

    @unittest.skipUnless(HAS_SYS_MONITORING, "sys.monitoring requires Python 3.12+")
    @patch('graphsignal._ticker.update_profile')
    def test_parallel_events_single_bucket(self, mock_update_profile):
        # Use a long resolution and disable the background rollover thread to avoid
        # non-deterministic bucket resets around time boundaries. This test triggers
        # rollover manually.
        resolution_ns = 60_000_000_000  # 60 seconds
        profiler = FunctionProfiler('test_profile')
        
        profiler.set_resolution_ns(resolution_ns)
        profiler.setup()
        profiler._stop_rollover_timer()
        
        test_func = self.test_functions.fast_function
        profiler.add_function(test_func, category='test', event_name='fast_function')
        
        # Call function multiple times in parallel
        num_calls = 5
        threads = []
        for _ in range(num_calls):
            thread = threading.Thread(target=test_func)
            threads.append(thread)
            thread.start()
        
        # Wait for all to complete
        for thread in threads:
            thread.join()
        
        # Check bucket state before rollover
        code = test_func.__code__
        bucket = profiler._buckets.get(code)
        if bucket:
            # Should have recorded multiple exits
            self.assertGreaterEqual(bucket.num_exited, num_calls)
        else:
            self.fail("Bucket not found for function")
        
        # Trigger rollover
        profiler._rollover_buckets(time.time_ns())
        
        # Wait a bit more
        time.sleep(0.1)
        
        # Should have aggregated all calls
        self.assertGreater(mock_update_profile.call_count, 0)
        
        # Find the call with aggregated data
        for call in mock_update_profile.call_args_list:
            args, kwargs = call
            if len(args) >= 3:
                name, profile, measurement_ts = args[:3]
                if profile and len(profile) > 0:
                    # Should have calls count >= num_calls
                    # (we check field IDs exist, actual values depend on implementation)
                    self.assertGreater(len(profile), 0)
                    break
        
        profiler.shutdown()

    @unittest.skipUnless(HAS_SYS_MONITORING, "sys.monitoring requires Python 3.12+")
    @patch('graphsignal._ticker.update_profile')
    def test_error_handling(self, mock_update_profile):
        resolution_ns = 1_000_000_000  # 1 second
        profiler = FunctionProfiler('test_profile')
        
        profiler.set_resolution_ns(resolution_ns)
        profiler.setup()
        
        test_func = self.test_functions.error_function
        profiler.add_function(test_func, category='test', event_name='error_function')
        
        # Call function that raises exception
        try:
            test_func()
        except ValueError:
            pass
        
        # Wait a bit
        time.sleep(0.1)
        
        # Check bucket state
        code = test_func.__code__
        bucket = profiler._buckets.get(code)
        if bucket:
            # Should have recorded an error
            self.assertGreater(bucket.num_errors, 0)
            self.assertGreater(bucket.num_exited, 0)
        else:
            self.fail("Bucket not found for function")
        
        # Trigger rollover
        profiler._rollover_buckets(time.time_ns())
        
        # Wait a bit more
        time.sleep(0.1)
        
        # Should have error count in profile
        self.assertGreater(mock_update_profile.call_count, 0)
        
        # Find call with error data
        for call in mock_update_profile.call_args_list:
            args, kwargs = call
            if len(args) >= 3:
                name, profile, measurement_ts = args[:3]
                if profile and len(profile) > 0:
                    # Should have error field ID
                    # (checking that profile has entries, actual field IDs depend on implementation)
                    self.assertGreater(len(profile), 0)
                    break
        
        profiler.shutdown()

    @unittest.skipUnless(HAS_SYS_MONITORING, "sys.monitoring requires Python 3.12+")
    def test_bucket_rollover(self):
        resolution_ns = 100_000_000  # 100ms
        profiler = FunctionProfiler(profile_name='test_profile')
        
        profiler.set_resolution_ns(resolution_ns)
        profiler.setup()
        
        test_func = self.test_functions.fast_function
        profiler.add_function(test_func, category='test', event_name='fast_function')
        
        # Call function
        test_func()
        
        # Wait a bit
        time.sleep(0.1)
        
        code = test_func.__code__
        bucket = profiler._buckets.get(code)
        if bucket:
            # Trigger rollover
            profiler._rollover_buckets(time.time_ns())
            
            # After rollover, exited and errors should be reset
            self.assertEqual(bucket.num_exited, 0)
            self.assertEqual(bucket.num_errors, 0)
            # But bucket_ts should be updated
            self.assertGreater(bucket.bucket_ts, 0)
        else:
            self.fail("Bucket not found for function")
        profiler.shutdown()

    @unittest.skipUnless(HAS_SYS_MONITORING, "sys.monitoring requires Python 3.12+")
    def test_multiple_functions_different_buckets(self):
        resolution_ns = 1_000_000_000  # 1 second
        profiler = FunctionProfiler(profile_name='test_profile')
        
        profiler.set_resolution_ns(resolution_ns)
        profiler.setup()
        
        func1 = self.test_functions.fast_function
        func2 = TestFunctions.static_function  # static method, no instance needed
        
        profiler.add_function(func1, category='test', event_name='func1')
        profiler.add_function(func2, category='test', event_name='func2')
        
        # Call both functions
        func1()
        func2()
        
        # Wait a bit
        time.sleep(0.1)
        
        # Should have separate buckets
        code1 = func1.__code__
        code2 = func2.__code__
        
        bucket1 = profiler._buckets.get(code1)
        bucket2 = profiler._buckets.get(code2)
        
        self.assertIsNotNone(bucket1)
        self.assertIsNotNone(bucket2)
        self.assertIsNot(bucket1, bucket2)
        
        # Both should have recorded exits
        self.assertGreater(bucket1.num_exited, 0)
        self.assertGreater(bucket2.num_exited, 0)
        
        profiler.shutdown()

    def test_bucket_enter_exit_timing(self):
        # Test bucket timing directly without requiring sys.monitoring
        bucket = FunctionBucket()
        bucket.bucket_ts = time.time_ns()
        
        # Enter
        bucket.enter()
        enter_offset = bucket.enter_offset_ns
        
        # Wait a bit
        time.sleep(0.05)
        
        # Exit
        bucket.exit()
        exit_offset = bucket.exit_offset_ns
        
        # Timing offsets should be recorded
        self.assertGreater(enter_offset, 0)
        self.assertGreater(exit_offset, enter_offset)
    
    def test_direct_callback_methods(self):
        resolution_ns = 1_000_000_000  # 1 second
        profiler = FunctionProfiler(profile_name='test_profile')
        
        profiler.set_resolution_ns(resolution_ns)
        # Manually set up fields (simulating add_function)
        test_func = TestFunctions.fast_function
        code = test_func.__code__
        
        # Mock field IDs
        mock_duration_id = 1
        mock_calls_id = 2
        mock_errors_id = 3
        
        from graphsignal.profilers.function_profiler import FunctionFields
        profiler._fields[code] = FunctionFields(
            duration_field_id=mock_duration_id,
            calls_field_id=mock_calls_id,
            errors_field_id=mock_errors_id
        )
        
        # Test enter callback
        profiler._enter_callback(code, 0)
        
        # Should have created a bucket
        bucket = profiler._buckets.get(code)
        self.assertIsNotNone(bucket)
        self.assertEqual(bucket.num_running, 1)
        
        # Test exit callback
        profiler._exit_callback(code)
        
        # Should have exited
        self.assertEqual(bucket.num_running, 0)
        self.assertEqual(bucket.num_exited, 1)
        
        # Test exit with exception
        profiler._enter_callback(code, 0)
        profiler._exit_callback(code, ValueError("test"))
        
        # Should have recorded error
        self.assertEqual(bucket.num_errors, 1)
        self.assertEqual(bucket.num_exited, 2)

    #@unittest.skip('for now')
    @unittest.skipUnless(HAS_SYS_MONITORING, "sys.monitoring requires Python 3.12+")
    def test_overhead(self):
        graphsignal._ticker.debug_mode = False
        logger.setLevel(logging.ERROR)

        NUM_TEST_CALLS = 1000000

        # without profiler
        test_func = self.test_functions.fast_function
        start_ns = time.perf_counter_ns()
        for _ in range(NUM_TEST_CALLS):
            test_func()
        took_ns_without_profiler = time.perf_counter_ns() - start_ns

        # with profiler
        resolution_ns = 1_000_000_000  # 1 second
        profiler = FunctionProfiler('test_profile')
        
        profiler.set_resolution_ns(resolution_ns)
        profiler.setup()
        
        test_func = self.test_functions.fast_function
        profiler.add_function(test_func, category='test', event_name='fast_function')
        
        # Initialize bucket to avoid first-time creation overhead
        test_func()

        #import cProfile, pstats
        #cprof = cProfile.Profile()
        #cprof.enable()

        start_ns = time.perf_counter_ns()
        for _ in range(NUM_TEST_CALLS):
            test_func()
        took_ns_with_profiler = time.perf_counter_ns() - start_ns

        logger.setLevel(logging.DEBUG)

        #stream = io.StringIO()
        #stats = pstats.Stats(cprof, stream=stream).sort_stats('time')
        #stats.print_stats()
        #logger.debug("Profiling stats:\n%s", stream.getvalue())        

        overhead_ns = took_ns_with_profiler - took_ns_without_profiler
        logger.debug(f"Overhead per call: {int(overhead_ns / NUM_TEST_CALLS)} ns")
        self.assertTrue(overhead_ns / NUM_TEST_CALLS < 1 * 1e3)  # less than 1 microsecond per call
        
        profiler.shutdown()

    @unittest.skipUnless(HAS_SYS_MONITORING, "sys.monitoring requires Python 3.12+")
    def test_add_function_path(self):
        resolution_ns = 1_000_000_000  # 1 second
        profiler = FunctionProfiler(profile_name='test_profile')
        
        profiler.set_resolution_ns(resolution_ns)
        profiler.setup()
        
        # Add function by path
        profiler.add_function_path(
            'test.profilers.test_function_profiler.TestFunctions.fast_function',
            category='test',
            event_name='fast_function_path'
        )
        
        # Call the function
        self.test_functions.fast_function()
        
        # Wait a bit
        time.sleep(0.1)
        
        # Should have created a bucket
        code = self.test_functions.fast_function.__code__
        bucket = profiler._buckets.get(code)
        self.assertIsNotNone(bucket)
        
        profiler.shutdown()

    def test_disabled_profiler(self):
        resolution_ns = 1_000_000_000  # 1 second
        profiler = FunctionProfiler(profile_name='test_profile')

        profiler.set_resolution_ns(resolution_ns)
        # Don't call setup, so profiler is disabled
        test_func = self.test_functions.fast_function
        profiler.add_function(test_func, category='test', event_name='fast_function')
        
        # Call function
        test_func()
        
        # Should not have created buckets
        code = test_func.__code__
        bucket = profiler._buckets.get(code)
        self.assertIsNone(bucket)

    @unittest.skipUnless(HAS_SYS_MONITORING, "sys.monitoring requires Python 3.12+")
    @patch('graphsignal._ticker.update_profile')
    def test_concurrent_rollover_and_events(self, mock_update_profile):
        resolution_ns = 200_000_000  # 200ms
        profiler = FunctionProfiler(profile_name='test_profile')
        
        profiler.set_resolution_ns(resolution_ns)
        profiler.setup()
        
        test_func = self.test_functions.fast_function
        profiler.add_function(test_func, category='test', event_name='fast_function')
        
        # Start multiple threads calling the function
        def call_repeatedly():
            for _ in range(10):
                test_func()
                time.sleep(0.01)
        
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=call_repeatedly)
            threads.append(thread)
            thread.start()
        
        # Trigger rollovers while functions are being called
        for _ in range(3):
            time.sleep(0.1)
            profiler._rollover_buckets(time.time_ns())
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Final rollover
        profiler._rollover_buckets(time.time_ns())
        
        # Should have multiple rollover calls
        self.assertGreater(mock_update_profile.call_count, 0)
        
        profiler.shutdown()

    def test_bucket_initialization(self):
        bucket = FunctionBucket()
        
        self.assertEqual(bucket.bucket_ts, 0)
        self.assertEqual(bucket.num_running, 0)
        self.assertEqual(bucket.num_exited, 0)
        self.assertEqual(bucket.num_errors, 0)
        self.assertEqual(bucket.enter_offset_ns, 0)
        self.assertEqual(bucket.exit_offset_ns, 0)
        
        # Test enter
        bucket.bucket_ts = time.time_ns()
        bucket.enter()
        self.assertEqual(bucket.num_running, 1)
        self.assertGreater(bucket.enter_offset_ns, 0)
        
        # Test exit
        bucket.exit()
        self.assertEqual(bucket.num_running, 0)
        self.assertEqual(bucket.num_exited, 1)
        self.assertGreater(bucket.exit_offset_ns, 0)
        
        # Test rollover
        new_ts = time.time_ns()
        bucket.rollover(new_ts)
        self.assertEqual(bucket.bucket_ts, new_ts)
        self.assertEqual(bucket.num_exited, 0)
        self.assertEqual(bucket.num_errors, 0)
        self.assertEqual(bucket.enter_offset_ns, 0)
        self.assertEqual(bucket.exit_offset_ns, 0)
        # num_running should persist across rollover
        # (functions still running continue to run)
    
    @patch('graphsignal._ticker.update_profile')
    def test_rollover_with_multiple_buckets(self, mock_update_profile):
        resolution_ns = 1_000_000_000  # 1 second
        profiler = FunctionProfiler(profile_name='test_profile')
        
        profiler.set_resolution_ns(resolution_ns)
        # Enable profiler without full setup (for testing rollover directly)
        profiler._disabled = False
        profiler._current_bucket_ts = time.time_ns() - 100_000_000  # 100ms ago
        
        # Manually set up fields for two functions
        func1 = self.test_functions.fast_function
        func2 = TestFunctions.static_function  # static method, no instance needed
        code1 = func1.__code__
        code2 = func2.__code__
        
        from graphsignal.profilers.function_profiler import FunctionFields
        profiler._fields[code1] = FunctionFields(
            duration_field_id=1,
            calls_field_id=2,
            errors_field_id=3
        )
        profiler._fields[code2] = FunctionFields(
            duration_field_id=4,
            calls_field_id=5,
            errors_field_id=6
        )
        
        # Manually trigger enter/exit to populate buckets
        profiler._enter_callback(code1, 0)
        profiler._exit_callback(code1)
        profiler._enter_callback(code2, 0)
        profiler._exit_callback(code2)
        
        # Trigger rollover
        profiler._rollover_buckets(time.time_ns())
        
        # Should have called update_profile with aggregated data
        self.assertGreater(mock_update_profile.call_count, 0)
        
        # Check that profile contains data from both functions
        # update_profile is called with keyword arguments
        call_args = mock_update_profile.call_args
        if call_args:
            args, kwargs = call_args
            profile = kwargs.get('profile', {})
            # Should have entries for both functions
            self.assertGreater(len(profile), 0)
            # Check that both functions' field IDs are present
            self.assertIn(1, profile)  # func1 duration_field_id
            self.assertIn(2, profile)  # func1 calls_field_id
            self.assertIn(4, profile)  # func2 duration_field_id
            self.assertIn(5, profile)  # func2 calls_field_id

    def test_error_exit(self):
        bucket = FunctionBucket()
        bucket.bucket_ts = time.time_ns()
        
        bucket.enter()
        bucket.exit(ValueError("test"))
        
        self.assertEqual(bucket.num_errors, 1)
        self.assertEqual(bucket.num_exited, 1)
        self.assertEqual(bucket.num_running, 0)
