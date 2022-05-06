import logging
import os
import sys
import time
import socket

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

    def read(self, profile):
        if not self._is_initialized:
            return

        device_count = nvmlDeviceGetCount()

        profile.node_usage.num_devices = device_count

        for i in range(0, device_count):
            try:
                handle = nvmlDeviceGetHandleByIndex(i)
            except NVMLError as err:
                log_nvml_error(err)
                continue

            device_usage = profile.device_usage.add()
            device_usage.device_type = profiles_pb2.DeviceType.GPU

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
                arch = nvmlDeviceGetArchitecture(handle)
                if arch == NVML_DEVICE_ARCH_KEPLER:
                    device_usage.architecture = 'Kepler'
                elif arch == NVML_DEVICE_ARCH_MAXWELL:
                    device_usage.architecture = 'Maxwell'
                elif arch == NVML_DEVICE_ARCH_PASCAL:
                    device_usage.architecture = 'Pascal'
                elif arch == NVML_DEVICE_ARCH_VOLTA:
                    device_usage.architecture = 'Volta'
                elif arch == NVML_DEVICE_ARCH_TURING:
                    device_usage.architecture = 'Turing'
                elif arch == NVML_DEVICE_ARCH_AMPERE:
                    device_usage.architecture = 'Ampere'
            except NVMLError as err:
                log_nvml_error(err)

            try:
                cc_major, cc_minor = nvmlDeviceGetCudaComputeCapability(handle)
                device_usage.compute_capability.major = cc_major
                device_usage.compute_capability.minor = cc_minor
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
                device_usage.pcie_throughput_tx = nvmlDeviceGetPcieThroughput(
                    handle, NVML_PCIE_UTIL_TX_BYTES)
                device_usage.pcie_throughput_rx = nvmlDeviceGetPcieThroughput(
                    handle, NVML_PCIE_UTIL_RX_BYTES)
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
