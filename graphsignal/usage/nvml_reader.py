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
    MIN_SAMPLE_READ_INTERVAL_US = int(10 * 1e6)

    def __init__(self):
        self._is_initialized = False
        self._last_nvlink_throughput_data_tx = {}
        self._last_nvlink_throughput_data_rx = {}

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

        now_us = int(time.time() * 1e6)

        device_count = nvmlDeviceGetCount()

        profile.node_usage.num_devices = device_count

        for idx in range(0, device_count):
            try:
                handle = nvmlDeviceGetHandleByIndex(idx)
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
                last_read_us = max(
                    int(graphsignal._agent.current_run.start_ms * 1e3),
                    now_us - NvmlReader.MIN_SAMPLE_READ_INTERVAL_US)

                sample_value_type, gpu_samples = nvmlDeviceGetSamples(handle, NVML_GPU_UTILIZATION_SAMPLES, last_read_us)
                device_usage.gpu_utilization_percent = _avg_sample_value(sample_value_type, gpu_samples)

                sample_value_type, mem_samples = nvmlDeviceGetSamples(handle, NVML_MEMORY_UTILIZATION_SAMPLES, last_read_us)
                device_usage.mem_access_percent = _avg_sample_value(sample_value_type, mem_samples)
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
                nvlink_throughput_data_tx = nvmlDeviceGetFieldValues(handle, [NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_TX])[0]
                if nvlink_throughput_data_tx.nvmlReturn == NVML_SUCCESS:
                    if idx in self._last_nvlink_throughput_data_tx:
                        last_data = self._last_nvlink_throughput_data_tx[idx]
                        interval_us = nvlink_throughput_data_tx.timestamp - last_data.timestamp
                        if interval_us > 0:
                            value = _nvml_value(nvlink_throughput_data_tx.valueType, nvlink_throughput_data_tx.value)
                            last_value = _nvml_value(last_data.valueType, last_data.value)
                            device_usage.nvlink_throughput_tx_kibs = (value - last_value) / (interval_us * 1e6)
                    self._last_nvlink_throughput_data_tx[idx] = nvlink_throughput_data_tx

                nvlink_throughput_data_rx = nvmlDeviceGetFieldValues(handle, [NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_RX])[0]
                if nvlink_throughput_data_rx.nvmlReturn == NVML_SUCCESS:
                    if idx in self._last_nvlink_throughput_data_rx:
                        last_data = self._last_nvlink_throughput_data_rx[idx]
                        interval_us = nvlink_throughput_data_rx.timestamp - last_data.timestamp
                        if interval_us > 0:
                            value = _nvml_value(nvlink_throughput_data_rx.valueType, nvlink_throughput_data_rx.value)
                            last_value = _nvml_value(last_data.valueType, last_data.value)
                            device_usage.nvlink_throughput_rx_kibs = (value - last_value) / (interval_us * 1e6)
                    self._last_nvlink_throughput_data_rx[idx] = nvlink_throughput_data_rx

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


def _nvml_value(value_type, value):
    if value_type == NVML_VALUE_TYPE_DOUBLE:
        return value.dVal
    if value_type == NVML_VALUE_TYPE_UNSIGNED_INT:
        return value.uiVal
    if value_type == NVML_VALUE_TYPE_UNSIGNED_LONG:
        return value.ulVal
    if value_type == NVML_VALUE_TYPE_UNSIGNED_LONG_LONG:
        return value.ullVal


def _avg_sample_value(sample_value_type, samples):
    if not samples:
        return 0

    sample_values = []

    if sample_value_type == NVML_VALUE_TYPE_DOUBLE:
        sample_values = [sample.sampleValue.dVal for sample in samples]
    if sample_value_type == NVML_VALUE_TYPE_UNSIGNED_INT:
        sample_values = [sample.sampleValue.uiVal for sample in samples]
    if sample_value_type == NVML_VALUE_TYPE_UNSIGNED_LONG:
        sample_values = [sample.sampleValue.ulVal for sample in samples]
    if sample_value_type == NVML_VALUE_TYPE_UNSIGNED_LONG_LONG:
        sample_values = [sample.sampleValue.ullVal for sample in samples]

    if len(sample_values) > 0:
        return sum(sample_values) / len(sample_values)

    return 0


def log_nvml_error(err):
    if (err.value == NVML_ERROR_NOT_SUPPORTED):
        logger.debug('NVML call not supported', exc_info=True)
    else:
        logger.error('Error calling NVML', exc_info=True)
