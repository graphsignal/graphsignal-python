import logging
import threading
import os
import random
import time
import traceback

import graphsignal
from graphsignal.profile_scheduler import ProfileScheduler
from graphsignal.proto import profiles_pb2

logger = logging.getLogger('graphsignal')


class InferenceStats:
    MAX_RESERVOIR_SIZE = 100
    MAX_COUNTERS = 10

    def __init__(self):
        self._update_lock = threading.Lock()
        self._start_sec = int(time.time())
        self.time_reservoir_us = []
        self.inference_counter = {}
        self.exception_counter = {}
        self.extra_counters = {}

    def add_time(self, duration_us):
        with self._update_lock:
            if len(self.time_reservoir_us) < InferenceStats.MAX_RESERVOIR_SIZE:
                self.time_reservoir_us.append(duration_us)
            else:
                self.time_reservoir_us[random.randint(0, InferenceStats.MAX_RESERVOIR_SIZE - 1)] = duration_us

    def inc_inference_counter(self, value, timestamp_us):
        with self._update_lock:
            self._inc_counter(self.inference_counter, value, timestamp_us)

    def inc_exception_counter(self, value, timestamp_us):
        with self._update_lock:
            self._inc_counter(self.exception_counter, value, timestamp_us)

    def inc_extra_counter(self, name, value, timestamp_us):
        with self._update_lock:
            if name not in self.extra_counters:
                if len(self.extra_counters) < InferenceStats.MAX_COUNTERS:
                    counter = self.extra_counters[name] = {}
                else:
                    return
            else:
                counter = self.extra_counters[name]

            self._inc_counter(counter, value, timestamp_us)

    def finalize(self, timestamp_us):
        with self._update_lock:
            self._finalize_counter(self.inference_counter, timestamp_us)
            self._finalize_counter(self.exception_counter, timestamp_us)
            for extra_counter in self.extra_counters.values():
                self._finalize_counter(extra_counter, timestamp_us)

    def _inc_counter(self, counter, value, timestamp_us):
        bucket = int(timestamp_us / 1e6)
        if bucket in counter:
            counter[bucket] += value
            counter[self._start_sec] = 0
        else:
            counter[self._start_sec] = 0
            counter[bucket] = value

    def _finalize_counter(self, counter, timestamp_us):
        end_sec = int(timestamp_us / 1e6)
        if end_sec not in counter:
            counter[end_sec] = 0


class Workload:
    MAX_EXCEPTIONS = 10

    def __init__(self):
        self.workload_id = None
        self.metrics = None
        self.profile_schedulers = {}
        self.inference_stats = {}
        self.exception_stats = {}

    def get_profile_scheduler(self, model_name):
        if model_name in self.profile_schedulers:
            return self.profile_schedulers[model_name]
        else:
            profile_scheduler = self.profile_schedulers[model_name] = ProfileScheduler()
            return profile_scheduler

    def reset_inference_stats(self, name):
        self.inference_stats[name] = InferenceStats()

    def get_inference_stats(self, name):
        if name not in self.inference_stats:
            self.reset_inference_stats(name)

        return self.inference_stats[name]

    def add_exception(self, exc_info, timestamp_us):
        if len(self.exception_stats) > Workload.MAX_EXCEPTIONS:
            return

        exc_type = ''
        if exc_info[0] and hasattr(exc_info[0], '__name__'):
            exc_type = str(exc_info[0].__name__)

        exc_message = ''
        if exc_info[1]:
            exc_message = str(exc_info[1])

        key = '{0}:{1}'.format(exc_type, exc_message)

        if key in self.exception_stats:
            exc_stat = self.exception_stats[key]
            exc_stat.count += 1
        else:
            exc_stat = self.exception_stats[key] = profiles_pb2.ExceptionStats()
            exc_stat.exc_type = exc_type
            exc_stat.message = exc_message
            if exc_info[2]:
                frames = traceback.format_tb(exc_info[2])
                if len(frames) > 0:
                    exc_stat.stack_trace = ''.join(frames)
            exc_stat.count += 1
        exc_stat.update_ms = int(timestamp_us / 1e3)

    def create_profile(self):
        profile = profiles_pb2.MLProfile()
        if graphsignal._agent.workload_name:
            profile.workload_name = graphsignal._agent.workload_name
        profile.worker_id = graphsignal._agent.worker_id
        profile.workload_id = self.workload_id
        return profile

    def upload(self, block=False):
        if block:
            graphsignal._agent.uploader.flush()
        else:
            graphsignal._agent.uploader.flush_in_thread()

    def tick(self, block=False):
        self.upload(block=False)

    def end(self, block=False):
        self.upload(block=block)
