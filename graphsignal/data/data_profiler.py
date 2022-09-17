import logging
import sys
import importlib
import functools

import graphsignal

logger = logging.getLogger('graphsignal')


class DataProfiler():
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

    def is_instance(self, data):
        raise NotImplementedError()

    def compute_counts(self, data):
        raise NotImplementedError()

    def build_stats(self, data):
        raise NotImplementedError()


def add_counts(data_stats, counts):
    for name, count in counts.items():
        dc = data_stats.data_counts.add()
        dc.name = name
        dc.count = count
