
_step_stats = {}

class StepStats:
    __slots__ = [
        'step_count',
        'sample_count',
        'total_time_us'
    ]

    def __init__(self):
        self.step_count = 0
        self.sample_count = 0
        self.total_time_us = 0


def reset_all_step_stats():
    global _step_stats
    _step_stats = {}


def init_step_stats(key):
    get_step_stats(key)


def reset_step_stats(key):
    del _step_stats[key]


def get_step_stats(key):
    if key in _step_stats:
        return _step_stats[key]
    else:
        stats = _step_stats[key] = StepStats()
        return stats


def update_step_stats(key, duration_us, effective_batch_size=None):
    stats = get_step_stats(key)
    stats.step_count += 1
    if effective_batch_size:
        stats.sample_count += effective_batch_size
    stats.total_time_us += duration_us

    return stats
