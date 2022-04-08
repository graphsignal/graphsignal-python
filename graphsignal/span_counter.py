

_span_stats = {}

class ProfilingSpanStats(object):
    __slots__ = [
        'count',
        'total_time_us',
    ]

    def __init__(self):
        self.count = 0
        self.total_time_us = 0


def reset_span_stats():
    global _span_stats
    _span_stats = {}


def get_span_stats(key):
    if key in _span_stats:
        return _span_stats[key]
    else:
        stats = _span_stats[key] = ProfilingSpanStats()
        return stats


def update_span_stats(key, total_time_us):
    stats = get_span_stats(key)
    stats.count += 1
    stats.total_time_us += total_time_us

    return stats