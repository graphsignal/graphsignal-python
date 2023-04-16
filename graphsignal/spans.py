import logging
import contextvars
import time


logger = logging.getLogger('graphsignal')

span_stack_var = contextvars.ContextVar('graphsignal_span_stack', default=[])


def push_current_span(span):
    span_stack_var.set(span_stack_var.get() + [span])


def pop_current_span():
    span_stack = span_stack_var.get()
    if len(span_stack) > 0:
        span_stack_var.set(span_stack[:-1])
        return span_stack[-1]
    return None


def get_current_span():
    span_stack = span_stack_var.get()
    if len(span_stack) > 0:
        return span_stack[-1]
    return None


def is_current_span(span):
    return get_current_span() == span


class Span:
    __slots__ = [
        'name',
        'start_ns',
        'end_ns',
        'trace_id',
        'children'
    ]

    MAX_CHILD_SPANS = 250

    def __init__(self, name, start_ns=None):
        self.name = name
        self.start_ns = start_ns if start_ns is not None else time.perf_counter_ns()
        self.end_ns = None
        self.trace_id = None
        self.children = None

    def stop(self, end_ns=None, trace_id=None):
        self.end_ns = end_ns if end_ns is not None else time.perf_counter_ns()
        self.trace_id = trace_id
    
    def set_trace_id(self, trace_id):
        self.trace_id = trace_id

    def add_child(self, child):
        if self.children is None:
            self.children = []
        if len(self.children) < self.MAX_CHILD_SPANS:
            self.children.append(child)


def start_span(name, start_ns=None):
    current_span = get_current_span()
    child_span = Span(name, start_ns=start_ns)
    push_current_span(child_span)
    if current_span:
        current_span.add_child(child_span)
    return child_span


def stop_span(end_ns=None, trace_id=None):
    current_span = pop_current_span()
    if current_span:
        current_span.stop(
            end_ns=end_ns, 
            trace_id=trace_id)
        return current_span

