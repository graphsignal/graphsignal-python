import logging
import contextvars
import time


logger = logging.getLogger('graphsignal')

span_stack_var = contextvars.ContextVar('span_stack', default=[])


def push_span(span):
    span_stack_var.set(span_stack_var.get() + [span])


def pop_span():
    span_stack = span_stack_var.get()
    if len(span_stack) > 0:
        span_stack_var.set(span_stack[:-1])
        return span_stack[-1]
    return None


def get_current_span():
    span_stack = span_stack_var.get()
    if len(span_stack) > 0:
        return span_stack[-1]


class TraceSpan:
    __slots__ = [
        'name',
        'start_ns',
        'end_ns',
        'has_exception',
        'children',
        'is_endpoint'
    ]

    def __init__(self, name, start_ns=None, is_endpoint=False):
        self.name = name
        self.start_ns = start_ns if start_ns is not None else time.perf_counter_ns()
        self.end_ns = None
        self.has_exception = None
        self.children = None
        self.is_endpoint = is_endpoint

    def stop(self, end_ns=None, has_exception=False):
        self.end_ns = end_ns if end_ns is not None else time.perf_counter_ns()
        self.has_exception = has_exception

    def add_child(self, child):
        if self.children is None:
            self.children = []
        self.children.append(child)


def start_span(name, start_ns=None, is_endpoint=False):
    current_span = get_current_span()
    child_span = TraceSpan(name, start_ns=start_ns, is_endpoint=is_endpoint)
    push_span(child_span)
    if current_span:
        current_span.add_child(child_span)
    return child_span


def stop_span(end_ns=None, has_exception=False):
    current_span = pop_span()
    if current_span:
        current_span.stop(end_ns=end_ns, has_exception=has_exception)
        return current_span


def is_current_span(span):
    span_stack = span_stack_var.get()
    if len(span_stack) > 0 and span_stack[-1] == span:
        return True
    return False
