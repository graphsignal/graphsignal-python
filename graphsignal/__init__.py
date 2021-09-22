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

UPLOAD_INTERVAL = 120

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
    '''
    Configures and initializes the logger.

    Args:
        api_key (:obj:`str`):
            The access key for communication with the Graphsignal servers.
        debug_mode (:obj:`bool`, optional):
            Enable/disable debug output.
        window_seconds (:obj:`int`, optional, default 300):
            The length of prediction time windows for which data statistics
            are reported.
        buffer_size (:obj:`int`, optional, default 100):
            The maximum mumber of model input and/or output data instances
            kept in memory before computing statistics.
    '''

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
    '''
    Send any collected data to the server and cleanup.

    Normally, when python scripts exists, this method is automatically called. Use this method,
    if you want to explicitely shutdown the logger.
    '''

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
    '''
    Check if any data can be uploaded and if yes, upload.
    Can be used in situations when logging is irregular and/or rare.
    '''

    if not _config:
        logger.error(
            'Logger not configured, please use graphsignal.configure()')
        return

    sessions.upload_all()
    _uploader.flush()


def session(deployment_name):
    '''
    Get logging session for the model identified by model deployment name.

    Args:
        deployment_name (:obj:`str`, optional):
            Model deployment name, e.g. `model1_production`, `modelB_canary` or any other string value.
    Returns:
        :obj:`Session` - session object for logging prediction data.
    Raises:
        `ValueError`: When `deployment_name` is invalid.
    '''

    if not _config:
        raise ValueError(
            'graphsignal not configured, please use graphsignal.configure()')

    return sessions.get_session(deployment_name)
