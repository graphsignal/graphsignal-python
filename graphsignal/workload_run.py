import logging
import threading
import os

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

        if 'GRAPHSIGNAL_TAGS' in os.environ:
            env_tags = os.environ['GRAPHSIGNAL_TAGS']
            if env_tags:
                for tag in env_tags.split(','):
                    self.add_tag(tag.strip())

        if 'GRAPHSIGNAL_PARAMS' in os.environ:
            env_params = os.environ['GRAPHSIGNAL_PARAMS']
            if env_params:
                for param in env_params.split(','):
                    pair = param.split(':')
                    if pair[0] and pair[1]:
                        self.add_param(pair[0].strip(), pair[1].strip())

    def add_tag(self, tag):
        if tag is None or not isinstance(tag, str):
            raise ValueError('add_tag: missing or invalid argument: tag')

        if self.tags is None:
            self.tags = {}
        self.tags[tag[:50]] = True

    def add_param(self, name, value):
        if self.params is None:
            self.params = {}
        self.params[name[:250]] = str(value)[:1000]

    def add_metric(self, name, value):
        if self.metrics is None:
            self.metrics = {}
        self.metrics[name[:250]] = value

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
