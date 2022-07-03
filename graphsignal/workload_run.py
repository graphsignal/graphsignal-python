

class StepStats:
    def __init__(self):
        self.step_count = 0
        self.sample_count = 0
        self.total_time_us = 0


class WorkloadRun:
    def __init__(self):
        self.start_ms = None
        self.run_id = None
        self.tags = None
        self.params = None
        self.metrics = None
        self.step_stats = {}

    def init_step_stats(self, key):
        self.get_step_stats(key)

    def reset_step_stats(self, key):
        del self.step_stats[key]

    def get_step_stats(self, key):
        if key in self.step_stats:
            return self.step_stats[key]
        else:
            stats = self.step_stats[key] = StepStats()
            return stats

    def update_step_stats(self, key, duration_us, effective_batch_size=None):
        stats = self.get_step_stats(key)
        stats.step_count += 1
        if effective_batch_size:
            stats.sample_count += effective_batch_size
        stats.total_time_us += duration_us

        return stats
