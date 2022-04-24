

_step_stats = {}

class ProfilingStepStats(object):
    __slots__ = [
        'count',
        'total_time_us',
    ]

    def __init__(self):
        self.count = 0
        self.total_time_us = 0


def reset_step_stats():
    global _step_stats
    _step_stats = {}


def get_step_stats(key):
    if key in _step_stats:
        return _step_stats[key]
    else:
        stats = _step_stats[key] = ProfilingStepStats()
        return stats


def update_step_stats(key, total_time_us):
    stats = get_step_stats(key)
    stats.count += 1
    stats.total_time_us += total_time_us

    return stats