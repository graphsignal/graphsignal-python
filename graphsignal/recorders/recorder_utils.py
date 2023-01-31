import logging
from functools import wraps
import asyncio

logger = logging.getLogger('graphsignal')

def patch_method(obj, func_name, before_func=None, after_func=None):
    if not hasattr(obj, func_name):
        return False

    func = getattr(obj, func_name)

    if hasattr(func, '__graphsignal_wrapped__'):
        return False

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
