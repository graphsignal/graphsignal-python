import logging
import contextvars
import time


logger = logging.getLogger('graphsignal')

span_stack_var = contextvars.ContextVar('graphsignal_span_stack', default=[])


def clear_span_stack():
    span_stack_var.set([])


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
        'operation',
        'trace_id',
        'parent_span',
        'root_span',
        'total_count',
        'in_context',
        'is_sampling'
    ]

    MAX_NESTED_SPANS = 250

    def __init__(self, operation):
        self.operation = operation
        self.trace_id = None
        self.root_span = None
        self.parent_span = None
        self.total_count = None
        self.in_context = False
        self.is_sampling = False

        parent_span = get_current_span()
        if not parent_span:
            self.init_as_root()
            push_current_span(self)
            self.in_context = True
        elif parent_span.can_add_child():
            self.init_as_child(parent_span)
            push_current_span(self)
            self.in_context = True

    def init_as_root(self):
        self.root_span = self
        self.total_count = 1
 
    def init_as_child(self, parent_span):
        self.parent_span = parent_span
        self.root_span = parent_span.root_span
        self.root_span.total_count += 1

    def set_trace_id(self, trace_id):
        self.trace_id = trace_id

    def set_sampling(self, is_sampling):
        self.is_sampling = is_sampling
    
    def is_root_sampling(self):
        return self.root_span and self.root_span.is_sampling

    def stop(self):
        if self.in_context:
            if is_current_span(self):
                pop_current_span()
            else:
                logger.error(f'Span.stop() called on a span that is not the current span {self.operation}')

    def can_add_child(self):
        return self.in_context and self.root_span.total_count < Span.MAX_NESTED_SPANS

    def __repr__(self):
        root_span_id = self.root_span.trace_id if self.root_span else None
        parent_span_id = self.parent_span.trace_id if self.parent_span else None
        return f'Span({self.operation}, {self.trace_id}, {root_span_id}, {parent_span_id}, {self.total_count}, {self.in_context}, {self.is_sampling})'