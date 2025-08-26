import logging
from functools import wraps
import asyncio
import types
import re
import time

import graphsignal

logger = logging.getLogger('graphsignal')

version_regexp = re.compile(r'^(\d+)\.?(\d+)?\.?(\d+)?')


def profile_method(obj, func_name, event_name=None, event_name_func=None, profile_func=None):
    def before_func(args, kwargs):
        profiled_event_name = None
        if event_name:
            profiled_event_name = event_name
        elif event_name_func:
            try:
                profiled_event_name = event_name_func(args, kwargs)
            except Exception as e:
                logger.debug('Error tracing %s', func_name, exc_info=True)
        else:
            profiled_event_name = func_name

        return dict(
            event_name=profiled_event_name,
            start_ns=time.perf_counter_ns())

    def after_func(args, kwargs, ret, exc, context):
        start_ns = context['start_ns']

        if not is_generator(ret) and not is_async_generator(ret):
            duration_ns = time.perf_counter_ns() - start_ns

            try:
                if not context.get('finished', False):
                    context['finished'] = True
                    profile_func(context['event_name'], duration_ns)
            except Exception as e:
                logger.debug('Error tracing %s', func_name, exc_info=True)

    def yield_func(stopped, item, context):
        start_ns = context['start_ns']

        if stopped:
            duration_ns = time.perf_counter_ns() - start_ns

            try:
                if not context.get('finished', False):
                    context['finished'] = True
                    profile_func(context['event_name'], duration_ns)
            except Exception as e:
                logger.debug('Error tracing %s', func_name, exc_info=True)

    if not patch_method(obj, func_name, before_func=before_func, after_func=after_func, yield_func=yield_func):
        logger.debug('Cannot instrument %s.', func_name)


def trace_method(obj, func_name, span_name=None, include_profiles=None, span_name_func=None, trace_func=None, data_func=None):
    def before_func(args, kwargs):
        if span_name_func is None:
            selected_span_name = span_name
        else:
            selected_span_name = span_name_func(args, kwargs)
        return dict(span=graphsignal.trace(span_name=selected_span_name, include_profiles=include_profiles))

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
            if data_func:
                data_func(span, item)

    if not patch_method(obj, func_name, before_func=before_func, after_func=after_func, yield_func=yield_func):
        logger.debug('Cannot instrument %s.', func_name)


def uninstrument_method(obj, func_name):
    if not unpatch_method(obj, func_name):
        logger.debug('Cannot uninstrument %s.', func_name)


class ObjectProxy:
    def __init__(self, obj):
        self._obj = obj

    def __getattr__(self, attr):
        return getattr(self._obj, attr)

    def __repr__(self):
        return repr(self._obj)


class GeneratorWrapper(ObjectProxy):
    def __init__(self, gen, yield_func, context=None):
        super().__init__(gen)
        self._gen = gen
        self._yield_func = yield_func
        self._context = context

    def __enter__(self):
        if hasattr(self._gen, '__enter__'):
            self._gen.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self._gen, '__exit__'):
            self._gen.__exit__(exc_type, exc_val, exc_tb)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            item = self._gen.__next__()
            try:
                self._yield_func(False, item, self._context)
            except:
                logger.debug('Exception in yield_func', exc_info=True)
            return item
        except StopIteration:
            try:
                self._yield_func(True, None, self._context)
            except:
                logger.debug('Exception in yield_func', exc_info=True)
            raise
        except Exception as e:
            try:
                self._yield_func(True, None, self._context)
            except:
                logger.debug('Exception in yield_func', exc_info=True)
            raise


class AsyncGeneratorWrapper(ObjectProxy):
    def __init__(self, async_gen, yield_func, context=None):
        super().__init__(async_gen)
        self._async_gen = async_gen
        self._yield_func = yield_func
        self._context = context

    async def __aenter__(self):
        if hasattr(self._async_gen, '__aenter__'):
            await self._async_gen.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self._async_gen, '__aexit__'):
            await self._async_gen.__aexit__(exc_type, exc_val, exc_tb)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            item = await self._async_gen.__anext__()
            try:
                self._yield_func(False, item, self._context)
            except:
                logger.debug('Exception in yield_func', exc_info=True)
            return item
        except StopAsyncIteration:
            try:
                self._yield_func(True, None, self._context)
            except:
                logger.debug('Exception in yield_func', exc_info=True)
            raise
        except Exception as e:
            try:
                self._yield_func(True, None, self._context)
            except:
                logger.debug('Exception in yield_func', exc_info=True)
            raise


def patch_method(obj, func_name, before_func=None, after_func=None, yield_func=None):
    if not hasattr(obj, func_name):
        return False

    func = getattr(obj, func_name)

    if hasattr(func, '__graphsignal_wrapped__'):
        return False

    if asyncio.iscoroutinefunction(func) or asyncio.iscoroutinefunction(getattr(func, '__wrapped__', None)):
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
                        ret = GeneratorWrapper(ret, yield_func, context)
                    elif is_async_generator(ret):
                        ret = AsyncGeneratorWrapper(ret, yield_func, context)
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
                        ret = GeneratorWrapper(ret, yield_func, context)
                    elif is_async_generator(ret):
                        ret = AsyncGeneratorWrapper(ret, yield_func, context)
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