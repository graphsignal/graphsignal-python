import logging
import threading
import os
import random

import graphsignal
from graphsignal.profile_scheduler import ProfileScheduler

logger = logging.getLogger('graphsignal')


class InferenceStats:
    MAX_RESERVOIR_SIZE = 100

    def __init__(self):
        self.inference_count = 0
        self.sample_count = 0
        self.total_time_us = 0
        self.time_samples_us = []

    def update(self, duration_us, batch_size=None):
        self.inference_count += 1
        if batch_size:
            self.sample_count += batch_size
        self.total_time_us += duration_us
        if len(self.time_samples_us) < InferenceStats.MAX_RESERVOIR_SIZE:
            self.time_samples_us.append(duration_us)
        else:
            self.time_samples_us[random.randint(0, InferenceStats.MAX_RESERVOIR_SIZE - 1)] = duration_us

    def inference_time_p95_us(self):
        num_time_samples = len(self.time_samples_us)
        if num_time_samples > 0:
            idx = int(num_time_samples * 95 / 100)
            return sorted(self.time_samples_us)[idx]
        return 0

    def inference_time_avg_us(self):
        if self.inference_count > 0 and self.total_time_us > 0:
            return self.total_time_us / self.inference_count
        return 0

    def inference_rate(self):
        if self.inference_count > 0 and self.total_time_us > 0:
            return self.inference_count / (self.total_time_us / 1e6)
        return 0

    def sample_rate(self):
        if self.sample_count > 0 and self.total_time_us > 0:
            return self.sample_count / (self.total_time_us / 1e6)
        return 0


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
        self.profile_scheduler = ProfileScheduler()

        self.total_inference_count = 0
        self.reset_inference_stats()

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

    def inc_total_inference_count(self, count=1):
        with self._update_lock:
            self.total_inference_count += count

    def reset_inference_stats(self):
        with self._update_lock:
            self.inference_stats = InferenceStats()

    def update_inference_stats(self, duration_us, batch_size=None):
        with self._update_lock:
            self.inference_stats.update(duration_us, batch_size=batch_size)

    def add_tag(self, tag):
        if tag is None or not isinstance(tag, str):
            raise ValueError('add_tag: missing or invalid argument: tag')

        if self.tags is None:
            self.tags = {}
        self.tags[tag[:50]] = True

        logger.debug('add_tag: %s', tag)

    def add_param(self, name, value):
        if self.params is None:
            self.params = {}
        self.params[name[:250]] = str(value)[:1000]

        logger.debug('add_param: %s=%s', name, value)

    def add_metric(self, name, value):
        if self.metrics is None:
            self.metrics = {}
        self.metrics[name[:250]] = value

        logger.debug('add_metric: %s=%f', name, value)

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
