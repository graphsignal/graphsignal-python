import logging
import threading
import os
import random
import time

import graphsignal
from graphsignal.profile_scheduler import ProfileScheduler
from graphsignal.proto import profiles_pb2

logger = logging.getLogger('graphsignal')


class FunctionStats:
    MAX_RESERVOIR_SIZE = 100
    MAX_COUNTERS = 10

    def __init__(self):
        self._update_lock = threading.Lock()
        self.time_reservoir_us = []
        self.inference_counter = {}
        self.exception_counter = {}
        self.extra_counters = {}

    def add_time(self, duration_us):
        with self._update_lock:
            if len(self.time_reservoir_us) < FunctionStats.MAX_RESERVOIR_SIZE:
                self.time_reservoir_us.append(duration_us)
            else:
                self.time_reservoir_us[random.randint(0, FunctionStats.MAX_RESERVOIR_SIZE - 1)] = duration_us

    def inc_inference_counter(self, value, timestamp_us):
        with self._update_lock:
            bucket = int(timestamp_us / 1e6)
            if bucket in self.inference_counter:
                self.inference_counter[bucket] += value
            else:
                self.inference_counter[bucket] = value

    def inc_exception_counter(self, value, timestamp_us):
        with self._update_lock:
            bucket = int(timestamp_us / 1e6)
            if bucket in self.exception_counter:
                self.exception_counter[bucket] += value
            else:
                self.exception_counter[bucket] = value

    def inc_extra_counter(self, name, value, timestamp_us):
        with self._update_lock:
            if name not in self.extra_counters:
                if len(self.extra_counters) < FunctionStats.MAX_COUNTERS:
                    counter = self.extra_counters[name] = {}
                else:
                    return
            else:
                counter = self.extra_counters[name]

            bucket = int(timestamp_us / 1e6)
            if bucket in counter:
                counter[bucket] += value
            else:
                counter[bucket] = value


class Workload:
    def __init__(self):
        self.workload_id = None
        self.metrics = None
        self.profile_schedulers = {}
        self.inference_stats = {}

    def get_profile_scheduler(self, model_name):
        if model_name in self.profile_schedulers:
            return self.profile_schedulers[model_name]
        else:
            profile_scheduler = self.profile_schedulers[model_name] = ProfileScheduler()
            return profile_scheduler

    def reset_inference_stats(self, name):
        self.inference_stats[name] = FunctionStats()

    def get_inference_stats(self, name):
        if name not in self.inference_stats:
            self.reset_inference_stats(name)

        return self.inference_stats[name]

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
