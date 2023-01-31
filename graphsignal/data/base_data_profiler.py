import logging
import sys
import importlib
from abc import ABC, abstractmethod
import functools

import graphsignal

logger = logging.getLogger('graphsignal')


class DataStats:
    __slots__ = ['type_name', 'shape', 'counts']

    def __init__(self, type_name=None, shape=None, counts=None):
        self.type_name = type_name
        self.shape = shape
        if counts is not None:
            self.counts = {name:value for name, value in counts.items() if value > 0}
        else:
            self.counts = {}

    def __repr__(self):
        return 'DataStats(type_name={0}, shape={1}, counts={2})'.format(
            self.type_name, self.shape, self.counts)


class BaseDataProfiler(ABC):
    def __init__(self):
        self._module = None
        self._is_checked = False

    def check_module(self, name):
        if not self._is_checked:
            self._is_checked = True
            if name in sys.modules:
                try:
                    self._module = importlib.import_module(name)
                except ImportError:
                    logger.error('Error importing {0}'.format(name))            
        return self._module

    @abstractmethod
    def is_instance(self, data):
        pass

    @abstractmethod
    def compute_stats(self, data):
        pass
