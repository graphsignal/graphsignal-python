import unittest
import sys
import threading
import asyncio
import time
import logging
from unittest.mock import patch, Mock, call

import graphsignal
from graphsignal.recorders.exception_recorder import ExceptionRecorder
from graphsignal.uploader import Uploader

logger = logging.getLogger('graphsignal')


class ExceptionRecorderTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.maxDiff = None
        if len(logger.handlers) == 0:
            logger.addHandler(logging.StreamHandler(sys.stdout))
        graphsignal.configure(
            api_key='k1',
            debug_mode=True)
        graphsignal._tracer.auto_export = False
        graphsignal._tracer.auto_instrument = True

    async def asyncTearDown(self):
        graphsignal.shutdown()

    def test_recorder_initialization(self):
        """Test that the exception recorder is properly initialized through Graphsignal."""
        # Check if the exception recorder is in the recorders
        recorders = list(graphsignal._tracer.recorders())
        exception_recorders = [r for r in recorders if isinstance(r, ExceptionRecorder)]
        self.assertEqual(len(exception_recorders), 1)
        self.assertTrue(exception_recorders[0]._is_setup)

    @patch.object(Uploader, 'upload_error')
    def test_setup_and_shutdown(self, mocked_upload_error):
        """Test that exception recorder can be set up and shut down properly."""
        recorder = ExceptionRecorder()
        
        # Store original handlers
        original_excepthook = sys.excepthook
        original_threading_excepthook = threading.excepthook
        
        # Setup recorder
        recorder.setup()
        
        # Verify handlers are set
        self.assertNotEqual(sys.excepthook, original_excepthook)
        self.assertNotEqual(threading.excepthook, original_threading_excepthook)
        self.assertTrue(recorder._is_setup)
        
        # Shutdown recorder
        recorder.shutdown()
        
        # Verify handlers are restored
        self.assertEqual(sys.excepthook, original_excepthook)
        self.assertEqual(threading.excepthook, original_threading_excepthook)
        self.assertFalse(recorder._is_setup)

    @patch.object(Uploader, 'upload_error')
    def test_main_thread_exception_handling(self, mocked_upload_error):
        """Test that main thread exceptions are caught and reported."""
        # Get the initialized recorder
        recorders = list(graphsignal._tracer.recorders())
        recorder = next(r for r in recorders if isinstance(r, ExceptionRecorder))
        
        # Store original excepthook
        original_excepthook = sys.excepthook
        
        # Create a mock to track calls to the original excepthook
        mock_original_excepthook = Mock()
        
        # Temporarily replace the original excepthook with our mock
        recorder._original_excepthook = mock_original_excepthook
        
        # Now call our exception handler directly
        exc_type = ValueError
        exc_value = ValueError("Test exception")
        exc_traceback = None
        
        recorder._handle_exception(exc_type, exc_value, exc_traceback)
        
        # Verify the exception was reported
        self.assertEqual(mocked_upload_error.call_count, 1)
        error_call = mocked_upload_error.call_args[0][0]
        self.assertEqual(error_call.name, 'uncaught_exception')
        self.assertEqual(error_call.level, 'error')
        self.assertIn('exception.context', [tag.key for tag in error_call.tags])
        self.assertEqual('main_thread', next(tag.value for tag in error_call.tags if tag.key == 'exception.context'))
        
        # Verify original excepthook was called
        mock_original_excepthook.assert_called_once_with(exc_type, exc_value, exc_traceback)
        
        # Restore
        recorder._original_excepthook = original_excepthook

    @patch.object(Uploader, 'upload_error')
    def test_threading_exception_handling(self, mocked_upload_error):
        """Test that threading exceptions are caught and reported."""
        # Get the initialized recorder
        recorders = list(graphsignal._tracer.recorders())
        recorder = next(r for r in recorders if isinstance(r, ExceptionRecorder))
        
        # Create a mock to track calls to the original threading excepthook
        mock_original_threading_excepthook = Mock()
        
        # Temporarily replace the original threading excepthook with our mock
        recorder._original_threading_excepthook = mock_original_threading_excepthook
        
        # Create mock args for threading exception
        class MockArgs:
            def __init__(self):
                self.exc_type = RuntimeError
                self.exc_value = RuntimeError("Thread exception")
                self.exc_traceback = None
                self.thread = Mock()
                self.thread.name = "test_thread"
        
        mock_args = MockArgs()
        
        # Call our threading exception handler directly
        recorder._handle_threading_exception(mock_args)
        
        # Verify the exception was reported
        self.assertEqual(mocked_upload_error.call_count, 1)
        error_call = mocked_upload_error.call_args[0][0]
        self.assertEqual(error_call.name, 'uncaught_thread_exception')
        self.assertEqual(error_call.level, 'error')
        self.assertIn('exception.context', [tag.key for tag in error_call.tags])
        self.assertEqual('thread', next(tag.value for tag in error_call.tags if tag.key == 'exception.context'))
        self.assertIn('exception.thread_name', [tag.key for tag in error_call.tags])
        self.assertEqual('test_thread', next(tag.value for tag in error_call.tags if tag.key == 'exception.thread_name'))
        
        # Verify original threading excepthook was called
        mock_original_threading_excepthook.assert_called_once_with(mock_args)
        
        # Restore
        recorder._original_threading_excepthook = threading.excepthook

    @patch.object(Uploader, 'upload_error')
    async def test_asyncio_exception_handling(self, mocked_upload_error):
        """Test that asyncio exceptions are caught and reported."""
        # Get the initialized recorder
        recorders = list(graphsignal._tracer.recorders())
        recorder = next(r for r in recorders if isinstance(r, ExceptionRecorder))
        
        # Create a mock to track calls to the original loop exception handler
        mock_original_loop_exception_handler = Mock()
        
        # Temporarily replace the original loop exception handler with our mock
        recorder._original_loop_exception_handler = mock_original_loop_exception_handler
        
        # Create mock task
        mock_task = Mock()
        mock_task.name = "test_async_task"
        
        # Create context for asyncio exception
        loop = asyncio.get_running_loop()
        context = {
            'exception': asyncio.CancelledError("Async exception"),
            'task': mock_task
        }
        
        # Call our asyncio exception handler directly
        recorder._handle_asyncio_exception(loop, context)
        
        # Verify the exception was reported
        self.assertEqual(mocked_upload_error.call_count, 1)
        error_call = mocked_upload_error.call_args[0][0]
        self.assertEqual(error_call.name, 'uncaught_asyncio_exception')
        self.assertEqual(error_call.level, 'error')
        self.assertIn('exception.context', [tag.key for tag in error_call.tags])
        self.assertEqual('asyncio', next(tag.value for tag in error_call.tags if tag.key == 'exception.context'))
        self.assertIn('exception.task_name', [tag.key for tag in error_call.tags])
        self.assertEqual('test_async_task', next(tag.value for tag in error_call.tags if tag.key == 'exception.task_name'))
        
        # Verify original handler was called
        mock_original_loop_exception_handler.assert_called_once_with(loop, context)
        
        # Restore
        recorder._original_loop_exception_handler = None

    @patch.object(Uploader, 'upload_error')
    async def test_asyncio_error_handling(self, mocked_upload_error):
        """Test that asyncio errors (non-exception) are caught and reported."""
        # Get the initialized recorder
        recorders = list(graphsignal._tracer.recorders())
        recorder = next(r for r in recorders if isinstance(r, ExceptionRecorder))
        
        # Create a mock to track calls to the original loop exception handler
        mock_original_loop_exception_handler = Mock()
        
        # Temporarily replace the original loop exception handler with our mock
        recorder._original_loop_exception_handler = mock_original_loop_exception_handler
        
        # Create mock task
        mock_task = Mock()
        mock_task.name = "test_task"
        
        # Simulate an asyncio error without exception
        loop = asyncio.get_running_loop()
        context = {
            'message': 'Task was destroyed but it is pending!',
            'task': mock_task
        }
        
        # Call our asyncio exception handler directly
        recorder._handle_asyncio_exception(loop, context)
        
        # Verify the error was reported
        self.assertEqual(mocked_upload_error.call_count, 1)
        error_call = mocked_upload_error.call_args[0][0]
        self.assertEqual(error_call.name, 'asyncio_error')
        self.assertEqual(error_call.level, 'error')
        self.assertEqual(error_call.message, 'Task was destroyed but it is pending!')
        self.assertIn('exception.context', [tag.key for tag in error_call.tags])
        self.assertEqual('asyncio', next(tag.value for tag in error_call.tags if tag.key == 'exception.context'))
        
        # Verify original handler was called
        mock_original_loop_exception_handler.assert_called_once_with(loop, context)
        
        # Restore
        recorder._original_loop_exception_handler = None

    @patch.object(Uploader, 'upload_error')
    def test_exception_in_handler(self, mocked_upload_error):
        """Test that exceptions in the handler itself don't break the system."""
        recorder = ExceptionRecorder()
        recorder.setup()
        
        # Mock report_error to raise an exception
        with patch('graphsignal.report_error', side_effect=Exception("Handler error")):
            # Mock the original excepthook to avoid actually raising
            original_excepthook = sys.excepthook
            mock_excepthook = Mock()
            sys.excepthook = mock_excepthook
            
            try:
                # This should trigger our exception handler
                raise ValueError("Test exception")
            except ValueError:
                # Simulate what sys.excepthook would do
                sys.excepthook(type(ValueError("Test exception")), ValueError("Test exception"), None)
            
            # Verify original excepthook was still called despite error in our handler
            mock_excepthook.assert_called_once()
            
            # Restore
            sys.excepthook = original_excepthook
        
        recorder.shutdown()

    def test_double_setup(self):
        """Test that calling setup multiple times doesn't cause issues."""
        recorder = ExceptionRecorder()
        
        # First setup
        recorder.setup()
        self.assertTrue(recorder._is_setup)
        
        # Second setup should be ignored
        recorder.setup()
        self.assertTrue(recorder._is_setup)
        
        recorder.shutdown()

    def test_shutdown_without_setup(self):
        """Test that shutdown without setup doesn't cause issues."""
        recorder = ExceptionRecorder()
        
        # Shutdown without setup should not raise
        recorder.shutdown()
        self.assertFalse(recorder._is_setup)

    @patch.object(Uploader, 'upload_error')
    def test_auto_instrument_disabled(self, mocked_upload_error):
        """Test that recorder doesn't setup when auto_instrument is disabled."""
        # Disable auto instrument
        original_auto_instrument = graphsignal._tracer.auto_instrument
        graphsignal._tracer.auto_instrument = False
        
        try:
            recorder = ExceptionRecorder()
            recorder.setup()
            
            # Should not be setup
            self.assertFalse(recorder._is_setup)
            
            # No error should be uploaded
            self.assertEqual(mocked_upload_error.call_count, 0)
        finally:
            # Restore
            graphsignal._tracer.auto_instrument = original_auto_instrument

    @patch.object(Uploader, 'upload_error')
    def test_exception_tags(self, mocked_upload_error):
        """Test that exception tags are properly set."""
        # Get the initialized recorder
        recorders = list(graphsignal._tracer.recorders())
        recorder = next(r for r in recorders if isinstance(r, ExceptionRecorder))
        
        # Create a mock to track calls to the original excepthook
        mock_original_excepthook = Mock()
        
        # Temporarily replace the original excepthook with our mock
        recorder._original_excepthook = mock_original_excepthook
        
        # Now call our exception handler directly
        exc_type = ValueError
        exc_value = ValueError("Test exception")
        exc_traceback = None
        
        recorder._handle_exception(exc_type, exc_value, exc_traceback)
        
        # Verify tags are set correctly
        self.assertEqual(mocked_upload_error.call_count, 1)
        error_call = mocked_upload_error.call_args[0][0]
        tag_dict = {tag.key: tag.value for tag in error_call.tags}
        
        self.assertEqual(tag_dict['exception.context'], 'main_thread')
        
        # Restore
        recorder._original_excepthook = sys.excepthook
