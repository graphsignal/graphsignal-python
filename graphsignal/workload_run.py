import logging
import threading

import graphsignal
from graphsignal.profile_scheduler import ProfileScheduler

logger = logging.getLogger('graphsignal')


class WorkloadRun:
    MAX_PROFILES = 25

    def __init__(self):
        self._update_lock = threading.Lock()
        self._profiles = []
        self.start_ms = None
        self.run_id = None
        self.tags = None
        self.params = None
        self.metrics = None
        self.inference_count = 0
        self.sample_count = 0
        self.total_time_us = 0
        self.profile_scheduler = ProfileScheduler()

    def update_inference_stats(self, duration_us, batch_size=None):
        with self._update_lock:
            self.inference_count += 1
            if batch_size:
                self.sample_count += batch_size
            self.total_time_us += duration_us

    def add_profile(self, profile):
        with self._update_lock:
            del self._profiles[0:-WorkloadRun.MAX_PROFILES]
            self._profiles.append(profile)

    def upload(self, block=False):
        with self._update_lock:
            if len(self._profiles) == 0:
                return
            outgoing_profiles = self._profiles
            self._profiles = []

        for profile in outgoing_profiles:
            if self.tags is not None:
                for value in self.tags.keys():
                    tag = profile.tags.add()
                    tag.value = value
            if self.params is not None:
                for name, value in self.params.items():
                    param = profile.params.add()
                    param.name = name
                    param.value = value
            if self.metrics is not None:
                for name, value in self.metrics.items():
                    metric = profile.metrics.add()
                    metric.name = name
                    metric.value = value            

            graphsignal._agent.uploader.upload_profile(profile)

        logger.debug('Uploading %d profiles', len(outgoing_profiles))

        if block:
            graphsignal._agent.uploader.flush()
        else:
            graphsignal._agent.uploader.flush_in_thread()

    def end(self, block=False):
        self.upload(block=block)
