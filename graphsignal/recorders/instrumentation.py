import logging
from functools import wraps
import asyncio
import types

import graphsignal
from graphsignal.span_context import TraceSpan

logger = logging.getLogger('graphsignal')


def instrument_method(obj, func_name, endpoint, trace_func):
    def before_func(args, kwargs):
        return dict(trace=graphsignal.start_trace(endpoint=endpoint))

    def after_func(args, kwargs, ret, exc, context):
        trace = context['trace']

        if not is_generator(ret):
            trace.measure()

        try:
            if exc is not None:
                trace.set_exception(exc)

            trace_func(trace, args, kwargs, ret, exc)
        except Exception as e:
            logger.debug('Error tracing %s', func_name, exc_info=True)

        if not is_generator(ret):
            trace.stop()

    def yield_func(idx, item, context):
        trace = context['trace']

        if idx == 0:
            context['span'] = TraceSpan('response')
        if idx == -1:
            if 'span' in context:
                span = context['span']
                span.stop()
                trace._root_span.add_child(span)
            trace.stop()

    if not patch_method(obj, func_name, before_func=before_func, after_func=after_func, yield_func=yield_func):
        logger.debug('Cannot instrument %s.', endpoint)


def uninstrument_method(obj, func_name, endpoint):
    if not unpatch_method(obj, func_name):
        logger.debug('Cannot uninstrument %s.', endpoint)


def patch_method(obj, func_name, before_func=None, after_func=None, yield_func=None):
    if not hasattr(obj, func_name):
        return False

    func = getattr(obj, func_name)

    if hasattr(func, '__graphsignal_wrapped__'):
        return False

    if yield_func:
        def gen_wrapper(gen, yield_func, context):
            for idx, item in enumerate(gen):
                try:
                    yield_func(idx, item, context)
                except:
                    logger.debug('Exception in yield_func', exc_info=True)
                yield item
            yield_func(-1, None, context)

    if asyncio.iscoroutinefunction(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            context = None
            exc = None
            ret = None

            if before_func:
                try:
                    context = before_func(args, kwargs)
                except:
                    logger.debug('Exception in before_func', exc_info=True)

            try:
                ret = await func(*args, **kwargs)
            except BaseException as e:
                exc = e

            if after_func:
                try:
                    after_func(args, kwargs, ret, exc, context)
                except:
                    logger.debug('Exception in after_func', exc_info=True)

            if yield_func:
                try:
                    if is_generator(ret):
                        ret = gen_wrapper(ret, yield_func, context)
                except:
                    logger.debug('Exception in yield_func', exc_info=True)

            if exc:
                raise exc
            return ret
    else:
        @wraps(func)
        def wrapper(*args, **kwargs):
            context = None
            exc = None
            ret = None

            if before_func:
                try:
                    context = before_func(args, kwargs)
                except:
                    logger.debug('Exception in before_func', exc_info=True)

            try:
                ret = func(*args, **kwargs)
            except BaseException as e:
                exc = e

            if after_func:
                try:
                    after_func(args, kwargs, ret, exc, context)
                except:
                    logger.debug('Exception in after_func', exc_info=True)

            if yield_func:
                try:
                    if is_generator(ret):
                        ret = gen_wrapper(ret, yield_func, context)
                except:
                    logger.debug('Exception in yield_func', exc_info=True)

            if exc:
                raise exc
            return ret

    setattr(wrapper, '__graphsignal_wrapped__', True)
    setattr(obj, func_name, wrapper)
    return True


def unpatch_method(obj, func_name):
    if not hasattr(obj, func_name):
        return False

    func = getattr(obj, func_name)

    if not hasattr(func, '__graphsignal_wrapped__'):
        return False

    if not hasattr(func, '__wrapped__'):
        return False
    
    setattr(obj, func_name, getattr(func, '__wrapped__'))
    return True


def is_generator(obj):
    return obj and isinstance(obj, types.GeneratorType)