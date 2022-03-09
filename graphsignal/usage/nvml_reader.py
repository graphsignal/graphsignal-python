import logging
import os
import sys
import time

import graphsignal
from graphsignal.proto import profiles_pb2
from graphsignal.usage.pynvml import *

logger = logging.getLogger('graphsignal')


class NvmlReader():
    __slots__ = [
        '_is_initialized'
    ]

    def __init__(self):
        self._is_initialized = False

    def setup(self):
        if self._is_initialized:
            return
        try:
            nvmlInit()
            self._is_initialized = True
            logger.debug('Initialized NVML')
        except BaseException:
            logger.debug('Error initializing NVML, skipping GPU usage')

    def shutdown(self):
        if not self._is_initialized:
            return
        try:
            nvmlShutdown()
            self._is_initialized = False
        except BaseException:
            logger.error('Error shutting down NVML', exc_info=True)

    def read(self, resource_usage):
        if not self._is_initialized:
            return

        device_count = nvmlDeviceGetCount()
        for i in range(0, device_count):
            device_usage = resource_usage.device_usage.add()
            handle = nvmlDeviceGetHandleByIndex(i)

            try:
                pci_info = nvmlDeviceGetPciInfo(handle)
                device_usage.device_id = pci_info.busId
            except NVMLError as err:
                log_nvml_error(err)

            try:
                device_usage.device_name = nvmlDeviceGetName(handle)
            except NVMLError as err:
                log_nvml_error(err)

            try:
                mem_info = nvmlDeviceGetMemoryInfo(handle)
                device_usage.mem_total = mem_info.total
                device_usage.mem_used = mem_info.used
                device_usage.mem_free = mem_info.total - mem_info.used
            except NVMLError as err:
                log_nvml_error(err)

            try:
                util_rates = nvmlDeviceGetUtilizationRates(handle)
                device_usage.gpu_utilization_percent = util_rates.gpu
                device_usage.mem_utilization_percent = util_rates.memory
            except NVMLError as err:
                log_nvml_error(err)

            try:
                device_usage.gpu_temp_c = nvmlDeviceGetTemperature(
                    handle, NVML_TEMPERATURE_GPU)
            except NVMLError as err:
                log_nvml_error(err)

            try:
                device_usage.power_usage_w = nvmlDeviceGetPowerUsage(
                    handle) / 1000.0
            except NVMLError as err:
                log_nvml_error(err)

            try:
                device_usage.fan_speed_percent = nvmlDeviceGetFanSpeed(
                    handle)
            except NVMLError as err:
                log_nvml_error(err)


def log_nvml_error(err):
    if (err.value == NVML_ERROR_NOT_SUPPORTED):
        logger.debug('NVML call not supported')
    else:
        logger.error('Error calling NVML', exc_info=True)
