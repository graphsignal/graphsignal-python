import logging
import os
import sys
import time

import graphsignal
from graphsignal.recorders.base_recorder import BaseRecorder
from graphsignal.vendor.pynvml.pynvml import *

logger = logging.getLogger('graphsignal')


class DeviceUsage:
    def __init__(self):
        self.device_type = None
        self.device_idx = None
        self.device_id = None
        self.device_name = None
        self.architecture = None
        self.compute_capability = None
        self.mem_total = 0
        self.mem_used = 0
        self.mem_free = 0
        self.mem_reserved = 0
        self.gpu_utilization_percent = 0
        self.mem_access_percent = 0
        self.pcie_throughput_tx = 0
        self.pcie_throughput_rx = 0
        self.nvlink_throughput_data_tx_kibs = 0
        self.nvlink_throughput_data_rx_kibs = 0
        self.gpu_temp_c = 0
        self.power_usage_w = 0
        self.fan_speed_percent = 0
        self.mxu_utilization_percent = 0
        self.processes = []
        self.drivers = []


class DeviceProcessUsage:
    def __init__(self):
        self.pid = None
        self.gpu_instance_id = None
        self.compute_instance_id = None
        self.mem_used = None


class DriverInfo:
    def __init__(self):
        self.name = None
        self.version = None


class SemVer:
    def __init__(self):
        self.major = 0
        self.minor = 0
        self.patch = 0


class NVMLRecorder(BaseRecorder):
    MIN_SAMPLE_READ_INTERVAL_US = int(10 * 1e6)

    def __init__(self):
        self._is_initialized = False
        self._setup_us = None
        self._last_nvlink_throughput_data_tx = {}
        self._last_nvlink_throughput_data_rx = {}
        self._last_snapshot = None

    def setup(self):
        try:
            nvmlInit()
            self._is_initialized = True
            logger.debug('Initialized NVML')
        except BaseException:
            logger.debug('Error initializing NVML, skipping GPU usage')

        self._setup_us = int(time.time() * 1e6)

        self.take_snapshot()

    def shutdown(self):
        if not self._is_initialized:
            return

        try:
            nvmlShutdown()
            self._is_initialized = False
        except BaseException:
            logger.error('Error shutting down NVML', exc_info=True)

    def on_metric_update(self):
        now = int(time.time())
        device_usages = self.take_snapshot()
        if len(device_usages) == 0:
            return

        for idx, device_usage in enumerate(device_usages):
            store = graphsignal._tracer.metric_store()
            metric_tags = {'deployment': graphsignal._tracer.deployment}
            if graphsignal._tracer.hostname:
                metric_tags['hostname'] = graphsignal._tracer.hostname
            metric_tags['device'] = idx

            if device_usage.gpu_utilization_percent > 0:
                store.set_gauge(
                    scope='system', name='gpu_utilization', tags=metric_tags, 
                    value=device_usage.gpu_utilization_percent, update_ts=now, unit='%')
            if device_usage.mxu_utilization_percent > 0:
                store.set_gauge(
                    scope='system', name='mxu_utilization', tags=metric_tags, 
                    value=device_usage.mxu_utilization_percent, update_ts=now, unit='%')
            if device_usage.mem_access_percent > 0:
                store.set_gauge(
                    scope='system', name='device_memory_access', tags=metric_tags, 
                    value=device_usage.mem_access_percent, update_ts=now, unit='%')
            if device_usage.mem_used > 0:
                store.set_gauge(
                    scope='system', name='device_memory_used', tags=metric_tags, 
                    value=device_usage.mem_used, update_ts=now, is_size=True)
            if device_usage.nvlink_throughput_data_tx_kibs > 0:
                store.set_gauge(
                    scope='system', name='nvlink_throughput_data_tx_kibs', tags=metric_tags, 
                    value=device_usage.nvlink_throughput_data_tx_kibs, update_ts=now, unit='KiB/s')
            if device_usage.nvlink_throughput_data_rx_kibs > 0:
                store.set_gauge(
                    scope='system', name='nvlink_throughput_data_rx_kibs', tags=metric_tags, 
                    value=device_usage.nvlink_throughput_data_rx_kibs, update_ts=now, unit='KiB/s')
            if device_usage.gpu_temp_c > 0:
                store.set_gauge(
                    scope='system', name='gpu_temp_c', tags=metric_tags, 
                    value=device_usage.gpu_temp_c, update_ts=now, unit='°C')
            if device_usage.power_usage_w > 0:
                store.set_gauge(
                    scope='system', name='power_usage_w', tags=metric_tags, 
                    value=device_usage.power_usage_w, update_ts=now, unit='W')

    def take_snapshot(self):
        if not self._is_initialized:
            return []

        device_usages = []

        now_us = int(time.time() * 1e6)

        device_count = nvmlDeviceGetCount()

        for idx in range(0, device_count):
            device_usage = DeviceUsage()
            device_usages.append(device_usage)

            try:
                version = nvmlSystemGetCudaDriverVersion_v2()
                if version:
                    driver_info = DriverInfo()
                    device_usage.drivers.append(driver_info)
                    driver_info.name = 'CUDA'
                    driver_info.version = _format_version(version)
            except NVMLError as err:
                _log_nvml_error(err)

            try:
                handle = nvmlDeviceGetHandleByIndex(idx)
            except NVMLError as err:
                _log_nvml_error(err)
                continue

            device_usage.device_idx = idx

            try:
                pci_info = nvmlDeviceGetPciInfo(handle)
                device_usage.device_id = pci_info.busId
            except NVMLError as err:
                _log_nvml_error(err)

            try:
                device_usage.device_name = nvmlDeviceGetName(handle)
            except NVMLError as err:
                _log_nvml_error(err)

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
                elif arch == NVML_DEVICE_ARCH_ADA:
                    device_usage.architecture = 'Ada'
                elif arch == NVML_DEVICE_ARCH_HOPPER:
                    device_usage.architecture = 'Hopper'
            except NVMLError as err:
                _log_nvml_error(err)

            try:
                cc_major, cc_minor = nvmlDeviceGetCudaComputeCapability(handle)
                device_usage.compute_capability = SemVer()
                device_usage.compute_capability.major = cc_major
                device_usage.compute_capability.minor = cc_minor
            except NVMLError as err:
                _log_nvml_error(err)

            try:
                mem_info = nvmlDeviceGetMemoryInfo_v2(handle)
                device_usage.mem_total = mem_info.total
                device_usage.mem_used = mem_info.used
                device_usage.mem_free = mem_info.total - mem_info.used
                device_usage.mem_reserved = mem_info.reserved
            except NVMLError as err:
                try:
                    mem_info = nvmlDeviceGetMemoryInfo(handle)
                    device_usage.mem_total = mem_info.total
                    device_usage.mem_used = mem_info.used
                    device_usage.mem_free = mem_info.total - mem_info.used
                except NVMLError as err:
                    _log_nvml_error(err)

            seen_pids = set()
            process_info_fns = [
                nvmlDeviceGetComputeRunningProcesses, 
                nvmlDeviceGetMPSComputeRunningProcesses, 
                nvmlDeviceGetGraphicsRunningProcesses]
            for process_info_fn in process_info_fns:
                try:
                    process_infos = process_info_fn(handle)
                    for process_info in process_infos:
                        if process_info.pid not in seen_pids:
                            seen_pids.add(process_info.pid)
                            device_process_usage = DeviceProcessUsage()
                            device_usage.processes.append(device_process_usage)
                            device_process_usage.pid = process_info.pid
                            device_process_usage.compute_instance_id = process_info.computeInstanceId
                            device_process_usage.gpu_instance_id = process_info.gpuInstanceId
                            if process_info.usedGpuMemory:
                                device_process_usage.mem_used = process_info.usedGpuMemory
                except NVMLError as err:
                    _log_nvml_error(err)

            try:
                last_read_us = max(
                    int(self._setup_us),
                    now_us - NVMLRecorder.MIN_SAMPLE_READ_INTERVAL_US)

                sample_value_type, gpu_samples = nvmlDeviceGetSamples(handle, NVML_GPU_UTILIZATION_SAMPLES, last_read_us)
                device_usage.gpu_utilization_percent = _avg_sample_value(sample_value_type, gpu_samples)

                sample_value_type, mem_samples = nvmlDeviceGetSamples(handle, NVML_MEMORY_UTILIZATION_SAMPLES, last_read_us)
                device_usage.mem_access_percent = _avg_sample_value(sample_value_type, mem_samples)
            except NVMLError as err:
                _log_nvml_error(err)

            try:
                device_usage.pcie_throughput_tx = nvmlDeviceGetPcieThroughput(
                    handle, NVML_PCIE_UTIL_TX_BYTES)
                device_usage.pcie_throughput_rx = nvmlDeviceGetPcieThroughput(
                    handle, NVML_PCIE_UTIL_RX_BYTES)
            except NVMLError as err:
                _log_nvml_error(err)

            try:
                nvlink_throughput_data_tx = nvmlDeviceGetFieldValues(handle, [NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_TX])[0]
                if nvlink_throughput_data_tx.nvmlReturn == NVML_SUCCESS:
                    value = _nvml_value(nvlink_throughput_data_tx.valueType, nvlink_throughput_data_tx.value)
                    if idx in self._last_nvlink_throughput_data_tx:
                        last_data = self._last_nvlink_throughput_data_tx[idx]
                        last_value = _nvml_value(last_data.valueType, last_data.value)
                        interval_us = nvlink_throughput_data_tx.timestamp - last_data.timestamp
                        if interval_us > 0:
                            device_usage.nvlink_throughput_data_tx_kibs = (value - last_value) / (interval_us * 1e6)
                    self._last_nvlink_throughput_data_tx[idx] = nvlink_throughput_data_tx

                nvlink_throughput_data_rx = nvmlDeviceGetFieldValues(handle, [NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_RX])[0]
                if nvlink_throughput_data_rx.nvmlReturn == NVML_SUCCESS:
                    value = _nvml_value(nvlink_throughput_data_rx.valueType, nvlink_throughput_data_rx.value)
                    if idx in self._last_nvlink_throughput_data_rx:
                        last_data = self._last_nvlink_throughput_data_rx[idx]
                        last_value = _nvml_value(last_data.valueType, last_data.value)
                        interval_us = nvlink_throughput_data_rx.timestamp - last_data.timestamp
                        if interval_us > 0:
                            device_usage.nvlink_throughput_data_rx_kibs = (value - last_value) / (interval_us * 1e6)
                    self._last_nvlink_throughput_data_rx[idx] = nvlink_throughput_data_rx
            except NVMLError as err:
                _log_nvml_error(err)

            try:
                device_usage.gpu_temp_c = nvmlDeviceGetTemperature(
                    handle, NVML_TEMPERATURE_GPU)
            except NVMLError as err:
                _log_nvml_error(err)

            try:
                device_usage.power_usage_w = nvmlDeviceGetPowerUsage(handle) / 1000.0
            except NVMLError as err:
                _log_nvml_error(err)

            try:
                device_usage.fan_speed_percent = nvmlDeviceGetFanSpeed(handle)
            except NVMLError as err:
                _log_nvml_error(err)

        self._last_snapshot = device_usages
        return device_usages


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


def _nvml_value(value_type, value):
    if value_type == NVML_VALUE_TYPE_DOUBLE:
        return value.dVal
    if value_type == NVML_VALUE_TYPE_UNSIGNED_INT:
        return value.uiVal
    if value_type == NVML_VALUE_TYPE_UNSIGNED_LONG:
        return value.ulVal
    if value_type == NVML_VALUE_TYPE_UNSIGNED_LONG_LONG:
        return value.ullVal


def _log_nvml_error(err):
    if (err.value == NVML_ERROR_NOT_SUPPORTED):
        logger.debug('NVML call not supported')
    elif (err.value == NVML_ERROR_NOT_FOUND):
        logger.debug('NVML call not found')
    else:
        logger.error('Error calling NVML', exc_info=True)


def _format_version(version):
    major = int(version / 1000)
    minor = int(version % 1000 / 10)
    return '{0}.{1}'.format(major, minor)
