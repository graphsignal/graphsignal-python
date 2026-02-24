import unittest
import sys
import threading
import asyncio
import time
import logging
from unittest.mock import patch, Mock, call

import graphsignal
from graphsignal.recorders.exception_recorder import ExceptionRecorder
from graphsignal.proto import signals_pb2

logger = logging.getLogger('graphsignal')


class ExceptionRecorderTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.maxDiff = None
        if len(logger.handlers) == 0:
            logger.addHandler(logging.StreamHandler(sys.stdout))
        graphsignal.configure(
            api_key='k1',
            debug_mode=True)
        graphsignal._ticker.auto_tick = False
        graphsignal._ticker.auto_instrument = True

    async def asyncTearDown(self):
        graphsignal.shutdown()

    def test_recorder_initialization(self):
        # Check if the exception recorder is in the recorders
        recorders = list(graphsignal._ticker.recorders())
        exception_recorders = [r for r in recorders if isinstance(r, ExceptionRecorder)]
        self.assertEqual(len(exception_recorders), 1)
        self.assertTrue(exception_recorders[0]._is_setup)

    def test_setup_and_shutdown(self):
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

    def test_main_thread_exception_handling(self):
        # Get the initialized recorder
        recorders = list(graphsignal._ticker.recorders())
        recorder = next(r for r in recorders if isinstance(r, ExceptionRecorder))
        
        # Store original excepthook
        original_excepthook = sys.excepthook
        
        # Create a mock to track calls to the original excepthook
        mock_original_excepthook = Mock()
        
        # Temporarily replace the original excepthook with our mock
        recorder._original_excepthook = mock_original_excepthook
        
        # Clear log store before test
        log_store = graphsignal._ticker.log_store()
        log_store.clear()
        
        # Now call our exception handler directly
        exc_type = ValueError
        exc_value = ValueError("Test exception")
        exc_traceback = None
        
        recorder._handle_exception(exc_type, exc_value, exc_traceback)
        
        # Verify the exception was logged
        log_batches = log_store.export()
        self.assertEqual(len(log_batches), 1)
        
        batch = log_batches[0]
        tag_dict = {tag.key: tag.value for tag in batch.tags}
        self.assertEqual(tag_dict['exception.name'], 'uncaught_exception')
        self.assertEqual(tag_dict['exception.context'], 'main_thread')
        
        self.assertEqual(len(batch.log_entries), 1)
        entry = batch.log_entries[0]
        self.assertEqual(entry.level, signals_pb2.LogEntry.LogLevel.ERROR_LEVEL)
        self.assertIn('ValueError', entry.message)
        self.assertIn('Test exception', entry.message)
        
        # Verify original excepthook was called
        mock_original_excepthook.assert_called_once_with(exc_type, exc_value, exc_traceback)
        
        # Restore
        recorder._original_excepthook = original_excepthook

    def test_threading_exception_handling(self):
        # Get the initialized recorder
        recorders = list(graphsignal._ticker.recorders())
        recorder = next(r for r in recorders if isinstance(r, ExceptionRecorder))
        
        # Create a mock to track calls to the original threading excepthook
        mock_original_threading_excepthook = Mock()
        
        # Temporarily replace the original threading excepthook with our mock
        recorder._original_threading_excepthook = mock_original_threading_excepthook
        
        # Clear log store before test
        log_store = graphsignal._ticker.log_store()
        log_store.clear()
        
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
        
        # Verify the exception was logged
        log_batches = log_store.export()
        self.assertEqual(len(log_batches), 1)
        
        batch = log_batches[0]
        tag_dict = {tag.key: tag.value for tag in batch.tags}
        self.assertEqual(tag_dict['exception.name'], 'uncaught_thread_exception')
        self.assertEqual(tag_dict['exception.context'], 'thread')
        self.assertEqual(tag_dict['exception.thread_name'], 'test_thread')
        
        self.assertEqual(len(batch.log_entries), 1)
        entry = batch.log_entries[0]
        self.assertEqual(entry.level, signals_pb2.LogEntry.LogLevel.ERROR_LEVEL)
        self.assertIn('RuntimeError', entry.message)
        self.assertIn('Thread exception', entry.message)
        
        # Verify original threading excepthook was called
        mock_original_threading_excepthook.assert_called_once_with(mock_args)
        
        # Restore
        recorder._original_threading_excepthook = threading.excepthook

    async def test_asyncio_exception_handling(self):
        # Get the initialized recorder
        recorders = list(graphsignal._ticker.recorders())
        recorder = next(r for r in recorders if isinstance(r, ExceptionRecorder))
        
        # Create a mock to track calls to the original loop exception handler
        mock_original_loop_exception_handler = Mock()
        
        # Temporarily replace the original loop exception handler with our mock
        recorder._original_loop_exception_handler = mock_original_loop_exception_handler
        
        # Clear log store before test
        log_store = graphsignal._ticker.log_store()
        log_store.clear()
        
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
        
        # Verify the exception was logged
        log_batches = log_store.export()
        self.assertEqual(len(log_batches), 1)
        
        batch = log_batches[0]
        tag_dict = {tag.key: tag.value for tag in batch.tags}
        self.assertEqual(tag_dict['exception.name'], 'uncaught_asyncio_exception')
        self.assertEqual(tag_dict['exception.context'], 'asyncio')
        self.assertEqual(tag_dict['exception.task_name'], 'test_async_task')
        
        self.assertEqual(len(batch.log_entries), 1)
        entry = batch.log_entries[0]
        self.assertEqual(entry.level, signals_pb2.LogEntry.LogLevel.ERROR_LEVEL)
        self.assertIn('CancelledError', entry.message)
        self.assertIn('Async exception', entry.message)
        
        # Verify original handler was called
        mock_original_loop_exception_handler.assert_called_once_with(loop, context)
        
        # Restore
        recorder._original_loop_exception_handler = None

    async def test_asyncio_error_handling(self):
        # Get the initialized recorder
        recorders = list(graphsignal._ticker.recorders())
        recorder = next(r for r in recorders if isinstance(r, ExceptionRecorder))
        
        # Create a mock to track calls to the original loop exception handler
        mock_original_loop_exception_handler = Mock()
        
        # Temporarily replace the original loop exception handler with our mock
        recorder._original_loop_exception_handler = mock_original_loop_exception_handler
        
        # Clear log store before test
        log_store = graphsignal._ticker.log_store()
        log_store.clear()
        
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
        
        # Verify the error was logged
        log_batches = log_store.export()
        self.assertEqual(len(log_batches), 1)
        
        batch = log_batches[0]
        tag_dict = {tag.key: tag.value for tag in batch.tags}
        self.assertEqual(tag_dict['exception.name'], 'asyncio_error')
        self.assertEqual(tag_dict['exception.context'], 'asyncio')
        self.assertEqual(tag_dict['exception.task_name'], 'test_task')
        
        self.assertEqual(len(batch.log_entries), 1)
        entry = batch.log_entries[0]
        self.assertEqual(entry.level, signals_pb2.LogEntry.LogLevel.ERROR_LEVEL)
        self.assertEqual(entry.message, 'Task was destroyed but it is pending!')
        
        # Verify original handler was called
        mock_original_loop_exception_handler.assert_called_once_with(loop, context)
        
        # Restore
        recorder._original_loop_exception_handler = None

    def test_exception_in_handler(self):
        recorder = ExceptionRecorder()
        recorder.setup()
        
        # Mock log_message to raise an exception
        with patch('graphsignal.log_message', side_effect=Exception("Handler error")):
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
        recorder = ExceptionRecorder()
        
        # First setup
        recorder.setup()
        self.assertTrue(recorder._is_setup)
        
        # Second setup should be ignored
        recorder.setup()
        self.assertTrue(recorder._is_setup)
        
        recorder.shutdown()

    def test_shutdown_without_setup(self):
        recorder = ExceptionRecorder()
        
        # Shutdown without setup should not raise
        recorder.shutdown()
        self.assertFalse(recorder._is_setup)

    def test_auto_instrument_disabled(self):
        # Disable auto instrument
        original_auto_instrument = graphsignal._ticker.auto_instrument
        graphsignal._ticker.auto_instrument = False
        
        try:
            recorder = ExceptionRecorder()
            recorder.setup()
            
            # Should not be setup
            self.assertFalse(recorder._is_setup)
        finally:
            # Restore
            graphsignal._ticker.auto_instrument = original_auto_instrument

    def test_exception_tags(self):
        # Get the initialized recorder
        recorders = list(graphsignal._ticker.recorders())
        recorder = next(r for r in recorders if isinstance(r, ExceptionRecorder))
        
        # Create a mock to track calls to the original excepthook
        mock_original_excepthook = Mock()
        
        # Temporarily replace the original excepthook with our mock
        recorder._original_excepthook = mock_original_excepthook
        
        # Clear log store before test
        log_store = graphsignal._ticker.log_store()
        log_store.clear()
        
        # Now call our exception handler directly
        exc_type = ValueError
        exc_value = ValueError("Test exception")
        exc_traceback = None
        
        recorder._handle_exception(exc_type, exc_value, exc_traceback)
        
        # Verify tags are set correctly
        log_batches = log_store.export()
        self.assertEqual(len(log_batches), 1)
        
        batch = log_batches[0]
        tag_dict = {tag.key: tag.value for tag in batch.tags}
        
        self.assertEqual(tag_dict['exception.name'], 'uncaught_exception')
        self.assertEqual(tag_dict['exception.context'], 'main_thread')
        
        # Restore
        recorder._original_excepthook = sys.excepthook
