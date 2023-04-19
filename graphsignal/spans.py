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


class SpanCounter:
    __slots__ = [
        'current_count'
    ]

    def __init__(self):
        self.current_count = 0

    def increment(self):
        self.current_count += 1


class Span:
    __slots__ = [
        'name',
        'start_ns',
        'end_ns',
        'trace_id',
        'children',
        'root_counter',
        'in_context'
    ]

    MAX_NESTED_SPANS = 250

    def __init__(self, name, start_ns=None):
        self.name = name
        self.start_ns = start_ns if start_ns is not None else time.perf_counter_ns()
        self.end_ns = None
        self.trace_id = None
        self.children = None
        self.root_counter = None
        self.in_context = False

        parent_span = get_current_span()

        if not parent_span:
            self.init_as_root()
            push_current_span(self)
            self.in_context = True
        elif parent_span.can_add_child():
            parent_span.add_child(self)
            push_current_span(self)
            self.in_context = True

    def stop(self, end_ns=None, trace_id=None):
        self.end_ns = end_ns if end_ns is not None else time.perf_counter_ns()
        self.trace_id = trace_id

        if self.in_context:
            if is_current_span(self):
                pop_current_span()
            else:
                logger.error(f'Span.stop() called on a span that is not the current span: {self.name}')

    def can_add_child(self):
        return self.in_context and self.total_count() < Span.MAX_NESTED_SPANS

    def set_trace_id(self, trace_id):
        self.trace_id = trace_id

    def total_count(self):
        return self.root_counter.current_count

    def init_as_root(self):
        self.root_counter = SpanCounter()
        self.root_counter.increment()

    def add_child(self, child):
        if self.children is None:
            self.children = []
        self.children.append(child)
        self.root_counter.increment()
        child.root_counter = self.root_counter
