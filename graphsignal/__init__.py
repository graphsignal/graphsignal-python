import time
import sys
import os
import logging
import threading
import uuid
import atexit

from graphsignal import version
from graphsignal import sessions
from graphsignal.uploader import Uploader

logger = logging.getLogger('graphsignal')

_config = None
_uploader = None


class Config(object):
    def __init__(self):
        self.api_key = None
        self.debug_mode = None
        self.window_seconds = None
        self.buffer_size = None


def _get_config():
    return _config


def _get_uploader():
    return _uploader


def configure(api_key, debug_mode=False, window_seconds=300, buffer_size=100):
    global _config, _uploader

    if _config:
        logger.warning('Logger already configured')
        return

    if debug_mode:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.WARNING)

    if not api_key:
        logger.error('Missing argument: api_key')
        return

    _config = Config()
    _config.api_key = api_key
    _config.debug_mode = debug_mode
    _config.window_seconds = window_seconds
    _config.buffer_size = buffer_size

    _uploader = Uploader()
    _uploader.configure()

    sessions.reset_all()

    atexit.register(shutdown)

    logger.debug('Logger configured')


def shutdown():
    global _config

    if not _config:
        logger.error(
            'Logger not configured, please use graphsignal.configure()')
        return

    atexit.unregister(shutdown)

    sessions.upload_all(force=True)
    _uploader.flush()

    _config = None

    logger.debug('Logger shutdown')


def tick():
    if not _config:
        logger.error(
            'Logger not configured, please use graphsignal.configure()')
        return

    sessions.upload_all()
    _uploader.flush()


def session(deployment_name):
    if not _config:
        raise ValueError(
            'graphsignal not configured, please use graphsignal.configure()')

    return sessions.get_session(deployment_name)
