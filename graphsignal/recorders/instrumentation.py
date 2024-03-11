import logging
from functools import wraps
import asyncio
import types
import re

import graphsignal

logger = logging.getLogger('graphsignal')

version_regexp = re.compile(r'^(\d+)\.?(\d+)?\.?(\d+)?')


def instrument_method(obj, func_name, op_name=None, op_func=None, trace_func=None, data_func=None):
    def before_func(args, kwargs):
        if op_name is None:
            operation = op_func(args, kwargs)
        else:
            operation = op_name
        return dict(span=graphsignal.trace(operation=operation))

    def after_func(args, kwargs, ret, exc, context):
        span = context['span']

        if not is_generator(ret) and not is_async_generator(ret):
            span.measure()

        try:
            if exc is not None:
                span.add_exception(exc)

            trace_func(span, args, kwargs, ret, exc)
        except Exception as e:
            logger.debug('Error tracing %s', func_name, exc_info=True)

        if not is_generator(ret) and not is_async_generator(ret):
            span.stop()

    def yield_func(stopped, item, context):
        span = context['span']

        if stopped:
            span.stop()
        else:
            span.first_token()
            if data_func:
                data_func(span, item)

    if not patch_method(obj, func_name, before_func=before_func, after_func=after_func, yield_func=yield_func):
        logger.debug('Cannot instrument %s.', func_name)


def uninstrument_method(obj, func_name):
    if not unpatch_method(obj, func_name):
        logger.debug('Cannot uninstrument %s.', func_name)


def patch_method(obj, func_name, before_func=None, after_func=None, yield_func=None):
    if not hasattr(obj, func_name):
        return False

    func = getattr(obj, func_name)

    if hasattr(func, '__graphsignal_wrapped__'):
        return False

    if asyncio.iscoroutinefunction(func) or asyncio.iscoroutinefunction(getattr(func, '__wrapped__', None)):
        if yield_func:
            async def async_generator_wrapper(gen, yield_func, context):
                async for item in gen:
                    try:
                        yield_func(False, item, context)
                    except:
                        logger.debug('Exception in yield_func', exc_info=True)
                    yield item
                yield_func(True, None, context)

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
                    if is_async_generator(ret):
                        ret = async_generator_wrapper(ret, yield_func, context)
                except:
                    logger.debug('Exception in yield_func', exc_info=True)

            if exc:
                raise exc
            return ret
    else:
        if yield_func:
            def generator_wrapper(gen, yield_func, context):
                for item in gen:
                    try:
                        yield_func(False, item, context)
                    except:
                        logger.debug('Exception in yield_func', exc_info=True)
                    yield item
                yield_func(True, None, context)

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
                        ret = generator_wrapper(ret, yield_func, context)
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
    if not obj:
        return False
    if isinstance(obj, types.GeneratorType):
        return True
    if hasattr(obj, '__iter__') and hasattr(obj, '__next__'):
        return True
    return False


def is_async_generator(obj):
    if not obj:
        return False
    if isinstance(obj, types.AsyncGeneratorType):
        return True
    if hasattr(obj, '__aiter__') and hasattr(obj, '__anext__'):
        return True
    return False


def read_args(args, kwargs, names):
    values = {}
    for name, arg in zip(names, args):
        values[name] = arg
    values.update(kwargs)
    return values


def parse_semver(version):
    parsed_version = [0, 0, 0]
    version_match = version_regexp.match(str(version))
    if version_match is not None:
        groups = version_match.groups()
        if groups[0] is not None:
            parsed_version[0] = int(groups[0])
        if groups[1] is not None:
            parsed_version[1] = int(groups[1])
        if groups[2] is not None:
            parsed_version[2] = int(groups[2])
    return tuple(parsed_version)


def compare_semver(version1, version2):
    version1_int = version1[0] * 1e6 + version1[1] * 1e3 + version1[2]
    version2_int = version2[0] * 1e6 + version2[1] * 1e3 + version2[2]
    if version1_int < version2_int:
        return -1
    if version1_int > version2_int:
        return 1
    else:
        return 0