import logging
import sys
import threading
import asyncio
import traceback
from typing import Optional, Any

import graphsignal
from graphsignal.recorders.base_recorder import BaseRecorder

logger = logging.getLogger('graphsignal')


class ExceptionRecorder(BaseRecorder):
    def __init__(self):
        self._original_excepthook = None
        self._original_threading_excepthook = None
        self._original_loop_exception_handler = None
        self._is_setup = False

    def setup(self):
        if not graphsignal._tracer.auto_instrument:
            return

        if self._is_setup:
            return

        # Store original exception handlers
        self._original_excepthook = sys.excepthook
        self._original_threading_excepthook = threading.excepthook

        # Set up main thread exception handler
        sys.excepthook = self._handle_exception

        # Set up threading exception handler
        threading.excepthook = self._handle_threading_exception

        # Set up asyncio exception handler for the current event loop
        try:
            loop = asyncio.get_running_loop()
            self._original_loop_exception_handler = loop.get_exception_handler()
            loop.set_exception_handler(self._handle_asyncio_exception)
        except RuntimeError:
            # No event loop running, will be set up when loop is created
            pass

        self._is_setup = True
        logger.debug('Exception recorder setup complete')

    def shutdown(self):
        if not self._is_setup:
            return

        # Restore original exception handlers
        if self._original_excepthook is not None:
            sys.excepthook = self._original_excepthook

        if self._original_threading_excepthook is not None:
            threading.excepthook = self._original_threading_excepthook

        # Restore asyncio exception handler
        try:
            loop = asyncio.get_running_loop()
            if self._original_loop_exception_handler is not None:
                loop.set_exception_handler(self._original_loop_exception_handler)
            else:
                loop.set_exception_handler(None)
        except RuntimeError:
            # No event loop running
            pass

        self._is_setup = False
        logger.debug('Exception recorder shutdown complete')

    def _handle_exception(self, exc_type, exc_value, exc_traceback):
        """Handle uncaught exceptions in the main thread."""
        try:
            # Report the exception
            self._report_exception(
                name='uncaught_exception',
                exc_info=(exc_type, exc_value, exc_traceback),
                context='main_thread'
            )
        except Exception as e:
            logger.error('Error in exception handler', exc_info=True)

        # Call the original exception handler
        if self._original_excepthook is not None:
            self._original_excepthook(exc_type, exc_value, exc_traceback)

    def _handle_threading_exception(self, args):
        """Handle uncaught exceptions in threads."""
        try:
            # Report the exception
            self._report_exception(
                name='uncaught_thread_exception',
                exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
                context='thread',
                thread_name=args.thread.name if hasattr(args, 'thread') and args.thread else None
            )
        except Exception as e:
            logger.error('Error in threading exception handler', exc_info=True)

        # Call the original exception handler
        if self._original_threading_excepthook is not None:
            self._original_threading_excepthook(args)

    def _handle_asyncio_exception(self, loop, context):
        """Handle uncaught exceptions in asyncio tasks."""
        try:
            exception = context.get('exception')
            if exception is not None:
                # Report the exception
                self._report_exception(
                    name='uncaught_asyncio_exception',
                    exc_info=(type(exception), exception, exception.__traceback__),
                    context='asyncio',
                    task_name=getattr(context.get('task'), 'name', None) if 'task' in context else None
                )
            else:
                # Handle other asyncio errors (e.g., Future exceptions)
                message = context.get('message', 'Unknown asyncio error')
                self._report_exception(
                    name='asyncio_error',
                    message=message,
                    context='asyncio',
                    task_name=getattr(context.get('task'), 'name', None) if 'task' in context else None
                )
        except Exception as e:
            logger.error('Error in asyncio exception handler', exc_info=True)

        # Call the original exception handler if it exists
        if self._original_loop_exception_handler is not None:
            self._original_loop_exception_handler(loop, context)

    def _report_exception(self, name: str, exc_info: Optional[tuple] = None, 
                         message: Optional[str] = None, context: Optional[str] = None,
                         thread_name: Optional[str] = None, task_name: Optional[str] = None):
        """Report an exception using graphsignal.report_error."""
        try:
            # Build tags
            tags = {}
            if context:
                tags['exception.context'] = context
            if thread_name:
                tags['exception.thread_name'] = thread_name
            if task_name:
                tags['exception.task_name'] = task_name

            # Report the error
            graphsignal.report_error(
                name=name,
                tags=tags,
                level='error',
                message=message,
                exc_info=exc_info
            )
        except Exception as e:
            logger.error('Error reporting exception', exc_info=True)
