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
        self.debug_mode = False
        self.log_instances = True
        self.log_system_metrics = True


def _get_config():
    return _config


def _get_uploader():
    return _uploader


def configure(api_key, debug_mode=False, log_instances=True,
              log_system_metrics=True):
    '''
    Configures and initializes the logger.

    Args:
        api_key (:obj:`str`):
            The access key for communication with the Graphsignal servers.
        debug_mode (:obj:`bool`, optional):
            Enable/disable debug output.
        log_instances (:obj:`bool`, optional, default is ``True``):
            Enable/disable the recording and uploading of data samples.
        log_system_metrics (:obj:`bool`, optional, default is ``True``):
            Enable/disable the recording and uploading of system metrics.
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
    _config.log_instances = log_instances
    _config.log_system_metrics = log_system_metrics

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


def session(model_name, deployment_name=None):
    '''
    Get logging session for the model identified by model name and optional model deployment name.

    Args:
        model_name (:obj:`str`):
            The name of the model.
        deployment_name (:obj:`str`, optional):
            Model deployment name, e.g. `production`, `canary` or any other string value.
    Returns:
        :obj:`Session` - session object for logging prediction data, metrics and events.
    Raises:
        `ValueError`: When `model_name` and/or `deployment_name` have wrong type or format.
    '''

    if not _config:
        raise ValueError(
            'graphsignal not configured, please use graphsignal.configure()')

    return sessions.get_session(model_name, deployment_name)
