import logging
import os
import sys
import time
import socket

import graphsignal
from graphsignal.recorders.base_recorder import BaseRecorder
from pynvml import *

# after pynvml import to avoid import conflicts
from typing import List, Dict, Optional, Any, Union

logger = logging.getLogger('graphsignal')


class DeviceUsage:
    def __init__(self):
        self.device_type: Optional[str] = None
        self.device_uuid: Optional[str] = None
        self.device_idx: Optional[int] = None
        self.bus_id: Optional[str] = None
        self.device_name: Optional[str] = None
        self.architecture: Optional[str] = None
        self.compute_capability: Optional['SemVer'] = None
        self.mem_total: int = 0
        self.mem_used: int = 0
        self.mem_free: int = 0
        self.mem_reserved: int = 0
        self.gpu_utilization_percent: float = 0.0
        self.mem_access_percent: float = 0.0
        
        # PCIe metrics
        self.pcie_throughput_tx: int = 0
        self.pcie_throughput_rx: int = 0
        self.pcie_utilization_tx_percent: float = 0.0
        self.pcie_utilization_rx_percent: float = 0.0
        self.pcie_bandwidth_tx_mbps: float = 0.0
        self.pcie_bandwidth_rx_mbps: float = 0.0
        self.pcie_replay_counter: int = 0
        self.pcie_gen: Optional[int] = None
        self.pcie_width: Optional[int] = None
        self.pcie_max_bandwidth_gbps: float = 0.0
        
        # NVLINK metrics
        self.nvlink_throughput_data_tx_kibs: float = 0.0
        self.nvlink_throughput_data_rx_kibs: float = 0.0
        self.nvlink_throughput_raw_tx_kibs: float = 0.0
        self.nvlink_throughput_raw_rx_kibs: float = 0.0
        self.nvlink_bandwidth_tx_gbps: float = 0.0
        self.nvlink_bandwidth_rx_gbps: float = 0.0
        self.nvlink_utilization_tx_percent: float = 0.0
        self.nvlink_utilization_rx_percent: float = 0.0
        self.nvlink_link_count: int = 0
        self.nvlink_active_links: int = 0
        self.nvlink_replay_errors: int = 0
        self.nvlink_recovery_errors: int = 0
        self.nvlink_crc_errors: int = 0
        self.nvlink_minor_errors: int = 0
        self.nvlink_major_errors: int = 0
        self.nvlink_fatal_errors: int = 0
        self.nvlink_link_speed_gbps: float = 0.0
        self.nvlink_link_width: int = 0
        
        # GPU metrics
        self.gpu_temp_c: int = 0
        self.power_usage_w: float = 0.0
        self.fan_speed_percent: int = 0
        self.mxu_utilization_percent: float = 0.0
        self.processes: List['DeviceProcessUsage'] = []
        self.drivers: List['DriverInfo'] = []
        
        # Error monitoring fields
        self.ecc_sbe_volatile_total: int = 0
        self.ecc_dbe_volatile_total: int = 0
        self.ecc_sbe_aggregate_total: int = 0
        self.ecc_dbe_aggregate_total: int = 0
        self.retired_pages_sbe: int = 0
        self.retired_pages_dbe: int = 0

        # XID error monitoring fields
        self.last_xid_error_codes: list[int] = []

    def __str__(self):
        processes_str = ', '.join(str(proc) for proc in self.processes)
        drivers_str = ', '.join(str(driver) for driver in self.drivers)

        return (
            f'DeviceUsage(device_type={self.device_type}, device_uuid={self.device_uuid}, device_idx={self.device_idx}, '
            f'bus_id={self.bus_id}, device_name={self.device_name}, architecture={self.architecture}, '
            f'compute_capability={self.compute_capability}, mem_total={self.mem_total}, mem_used={self.mem_used}, '
            f'mem_free={self.mem_free}, mem_reserved={self.mem_reserved}, gpu_utilization_percent={self.gpu_utilization_percent}, '
            f'mem_access_percent={self.mem_access_percent}, pcie_throughput_tx={self.pcie_throughput_tx}, '
            f'pcie_throughput_rx={self.pcie_throughput_rx}, pcie_utilization_tx_percent={self.pcie_utilization_tx_percent}, '
            f'pcie_utilization_rx_percent={self.pcie_utilization_rx_percent}, pcie_bandwidth_tx_mbps={self.pcie_bandwidth_tx_mbps}, '
            f'pcie_bandwidth_rx_mbps={self.pcie_bandwidth_rx_mbps}, pcie_replay_counter={self.pcie_replay_counter}, '
            f'pcie_gen={self.pcie_gen}, pcie_width={self.pcie_width}, pcie_max_bandwidth_gbps={self.pcie_max_bandwidth_gbps}, '
            f'nvlink_throughput_data_tx_kibs={self.nvlink_throughput_data_tx_kibs}, '
            f'nvlink_throughput_data_rx_kibs={self.nvlink_throughput_data_rx_kibs}, '
            f'nvlink_throughput_raw_tx_kibs={self.nvlink_throughput_raw_tx_kibs}, '
            f'nvlink_throughput_raw_rx_kibs={self.nvlink_throughput_raw_rx_kibs}, '
            f'nvlink_bandwidth_tx_gbps={self.nvlink_bandwidth_tx_gbps}, nvlink_bandwidth_rx_gbps={self.nvlink_bandwidth_rx_gbps}, '
            f'nvlink_utilization_tx_percent={self.nvlink_utilization_tx_percent}, nvlink_utilization_rx_percent={self.nvlink_utilization_rx_percent}, '
            f'nvlink_link_count={self.nvlink_link_count}, nvlink_active_links={self.nvlink_active_links}, '
            f'nvlink_replay_errors={self.nvlink_replay_errors}, nvlink_recovery_errors={self.nvlink_recovery_errors}, '
            f'nvlink_crc_errors={self.nvlink_crc_errors}, nvlink_minor_errors={self.nvlink_minor_errors}, '
            f'nvlink_major_errors={self.nvlink_major_errors}, nvlink_fatal_errors={self.nvlink_fatal_errors}, '
            f'nvlink_link_speed_gbps={self.nvlink_link_speed_gbps}, nvlink_link_width={self.nvlink_link_width}, '
            f'gpu_temp_c={self.gpu_temp_c}, power_usage_w={self.power_usage_w}, fan_speed_percent={self.fan_speed_percent}, '
            f'mxu_utilization_percent={self.mxu_utilization_percent}, processes=[{processes_str}], drivers=[{drivers_str}], '
            f'ecc_sbe_volatile_total={self.ecc_sbe_volatile_total}, ecc_dbe_volatile_total={self.ecc_dbe_volatile_total}, '
            f'ecc_sbe_aggregate_total={self.ecc_sbe_aggregate_total}, ecc_dbe_aggregate_total={self.ecc_dbe_aggregate_total}, '
            f'retired_pages_sbe={self.retired_pages_sbe}, retired_pages_dbe={self.retired_pages_dbe}, xid_errors={self.last_xid_error_codes})'
        )

class DeviceProcessUsage:
    def __init__(self):
        self.pid: Optional[int] = None
        self.gpu_instance_id: Optional[int] = None
        self.compute_instance_id: Optional[int] = None
        self.mem_used: Optional[int] = None

    def __str__(self):
        return (
            f'DeviceProcessUsage(pid={self.pid}, gpu_instance_id={self.gpu_instance_id}, '
            f'compute_instance_id={self.compute_instance_id}, mem_used={self.mem_used})'
        )

class DriverInfo:
    def __init__(self):
        self.name: Optional[str] = None
        self.version: Optional[str] = None
    
    def __str__(self):
        return f'DriverInfo(name={self.name}, version={self.version})'


class SemVer:
    def __init__(self):
        self.major: int = 0
        self.minor: int = 0
        self.patch: int = 0


class NVMLRecorder(BaseRecorder):
    MIN_SAMPLE_READ_INTERVAL_US = int(10 * 1e6)

    def __init__(self):
        self._is_initialized: bool = False
        self._setup_us: Optional[int] = None
        self._last_nvlink_throughput_data_tx: Dict[int, Any] = {}
        self._last_nvlink_throughput_data_rx: Dict[int, Any] = {}
        self._last_nvlink_throughput_raw_tx: Dict[int, Any] = {}
        self._last_nvlink_throughput_raw_rx: Dict[int, Any] = {}
        self._last_pcie_throughput_tx: Dict[int, Any] = {}
        self._last_pcie_throughput_rx: Dict[int, Any] = {}
        self._last_snapshot: Optional[List[DeviceUsage]] = None
        
        # Error monitoring
        self._event_sets: Dict[int, Any] = {}  # device_idx -> event_set
        self._error_counters: Dict[int, Dict[str, Any]] = {}  # device_idx -> error counters

        self._hostname: Optional[str] = None
        try:
            self._hostname = socket.gethostname()
        except BaseException:
            logger.debug('Error reading hostname', exc_info=True)

    def setup(self):
        try:
            nvmlInit()
            self._is_initialized = True
            logger.debug('Initialized NVML')
        except BaseException:
            logger.debug('Error initializing NVML, skipping GPU usage')

        self._setup_us = int(time.time() * 1e6)

        device_usages = self.take_snapshot()

        if len(device_usages) > 0:
            cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
            if cuda_visible_devices:
                visible_idxs = list(map(int, cuda_visible_devices.split(',')))
            else:
                visible_idxs = list(range(len(device_usages)))  # all devices visible
            
            local_rank = None
            for env_var in ["LOCAL_RANK", "SLURM_LOCALID", "NCCL_LOCAL_RANK", "OMPI_COMM_WORLD_LOCAL_RANK"]:
                if env_var in os.environ:
                    local_rank = os.environ[env_var]
                    break
            if local_rank:
                default_idx = int(local_rank)
            else:
                default_idx = visible_idxs[0]

            tracer = graphsignal._tracer
            for device_usage in device_usages:
                if device_usage.device_idx == default_idx:
                    if device_usage.bus_id:
                        tracer.set_tag('device.bus_id', device_usage.bus_id)
                    if device_usage.device_uuid:
                        tracer.set_tag('device.uuid', device_usage.device_uuid)
                    if self._hostname and device_usage.bus_id:
                        tracer.set_tag('device.address', f'{self._hostname}:{device_usage.bus_id}')
                    if device_usage.device_name:
                        tracer.set_tag('device.name', device_usage.device_name)
                    break

        # Setup error monitoring
        self._setup_error_monitoring()

    def _setup_error_monitoring(self):
        if not self._is_initialized:
            return
            
        try:
            device_count = nvmlDeviceGetCount()
            for idx in range(device_count):
                try:
                    handle = nvmlDeviceGetHandleByIndex(idx)
                    
                    # Create event set for this device
                    event_set = nvmlEventSetCreate()
                    self._event_sets[idx] = event_set
                    
                    # Initialize error counters
                    self._error_counters[idx] = {
                        'last_xid_error_codes': []
                    }
                    
                    # Register for XID critical errors
                    try:
                        nvmlDeviceRegisterEvents(handle, nvmlEventTypeXidCriticalError, event_set)
                        logger.debug(f'Registered for XID events on device {idx}')
                    except Exception as err:
                        if str(err) == "Not Supported":
                            logger.debug(f'XID event monitoring not supported on device {idx}')
                        else:
                            logger.warning(f'Failed to register for XID events on device {idx}: {err}')
                            
                except Exception as err:
                    logger.warning(f'Failed to setup error monitoring for device {idx}: {err}')
                    
        except Exception as e:
            logger.warning(f'Error setting up error monitoring: {e}')

    def _check_for_errors(self):
        for device_idx, event_set in self._event_sets.items():
            try:
                # Non-blocking check for events (timeout=0)
                try:
                    event = nvmlEventSetWait_v2(event_set, 0)
                except Exception as err:
                    event = nvmlEventSetWait(event_set, 0)
                
                if event.eventType & nvmlEventTypeXidCriticalError:
                    error_code = event.eventData
                    self._error_counters[device_idx]['last_xid_error_codes'].append(error_code)
            except Exception as err:
                if hasattr(err, 'value') and err.value == NVML_ERROR_TIMEOUT:
                    pass
                else:
                    _log_nvml_error(err)

    def shutdown(self):
        if not self._is_initialized:
            return

        # Clean up event sets
        for device_idx, event_set in self._event_sets.items():
            try:
                nvmlEventSetFree(event_set)
                logger.debug(f'Freed event set for device {device_idx}')
            except Exception as err:
                _log_nvml_error(err)
        
        self._event_sets.clear()
        self._error_counters.clear()

        try:
            nvmlShutdown()
            self._is_initialized = False
        except BaseException:
            logger.error('Error shutting down NVML', exc_info=True)

    def on_metric_update(self):
        now = int(time.time())
        
        self._check_for_errors()
        
        device_usages = self.take_snapshot()

        logger.debug('Reading device usage: %s', '\n'.join(str(du) for du in device_usages))

        if len(device_usages) == 0:
            return

        for idx, device_usage in enumerate(device_usages):
            store = graphsignal._tracer.metric_store()
            metric_tags = graphsignal._tracer.tags.copy()
            if device_usage.bus_id:
                metric_tags['device.bus_id'] = device_usage.bus_id
            if device_usage.device_uuid:
                metric_tags['device.uuid'] = device_usage.device_uuid
            if self._hostname and device_usage.bus_id:
                metric_tags['device.address'] = f'{self._hostname}:{device_usage.bus_id}'
            if device_usage.device_name:
                metric_tags['device.name'] = device_usage.device_name

            if device_usage.gpu_utilization_percent > 0:
                store.set_gauge(
                    name='gpu.utilization', tags=metric_tags, 
                    value=device_usage.gpu_utilization_percent, update_ts=now, unit='percent')
            if device_usage.mxu_utilization_percent > 0:
                store.set_gauge(
                    name='gpu.mxu.utilization', tags=metric_tags, 
                    value=device_usage.mxu_utilization_percent, update_ts=now, unit='percent')
            if device_usage.mem_access_percent > 0:
                store.set_gauge(
                    name='gpu.memory.access', tags=metric_tags, 
                    value=device_usage.mem_access_percent, update_ts=now, unit='percent')
            if device_usage.mem_used > 0:
                store.set_gauge(
                    name='gpu.memory.usage', tags=metric_tags, 
                    value=device_usage.mem_used, update_ts=now, is_size=True)
            if device_usage.mem_free > 0:
                store.set_gauge(
                    name='gpu.memory.free', tags=metric_tags, 
                    value=device_usage.mem_free, update_ts=now, is_size=True)
            if device_usage.mem_total > 0:
                store.set_gauge(
                    name='gpu.memory.total', tags=metric_tags, 
                    value=device_usage.mem_total, update_ts=now, is_size=True)
            if device_usage.mem_reserved > 0:
                store.set_gauge(
                    name='gpu.memory.reserved', tags=metric_tags, 
                    value=device_usage.mem_reserved, update_ts=now, is_size=True)
            if device_usage.gpu_temp_c > 0:
                store.set_gauge(
                    name='gpu.temperature', tags=metric_tags, 
                    value=device_usage.gpu_temp_c, update_ts=now, unit='celsius')
            if device_usage.power_usage_w > 0:
                store.set_gauge(
                    name='gpu.power.usage', tags=metric_tags, 
                    value=device_usage.power_usage_w, update_ts=now, unit='watts')
            
            # PCIe metrics
            if device_usage.pcie_throughput_tx > 0:
                store.set_gauge(
                    name='gpu.pcie.throughput.tx', tags=metric_tags, 
                    value=device_usage.pcie_throughput_tx, update_ts=now, unit='bytes_per_second')
            if device_usage.pcie_throughput_rx > 0:
                store.set_gauge(
                    name='gpu.pcie.throughput.rx', tags=metric_tags, 
                    value=device_usage.pcie_throughput_rx, update_ts=now, unit='bytes_per_second')
            if device_usage.pcie_utilization_tx_percent > 0:
                store.set_gauge(
                    name='gpu.pcie.utilization.tx', tags=metric_tags, 
                    value=device_usage.pcie_utilization_tx_percent, update_ts=now, unit='percent')
            if device_usage.pcie_utilization_rx_percent > 0:
                store.set_gauge(
                    name='gpu.pcie.utilization.rx', tags=metric_tags, 
                    value=device_usage.pcie_utilization_rx_percent, update_ts=now, unit='percent')
            if device_usage.pcie_bandwidth_tx_mbps > 0:
                store.set_gauge(
                    name='gpu.pcie.bandwidth.tx', tags=metric_tags, 
                    value=device_usage.pcie_bandwidth_tx_mbps, update_ts=now, unit='megabits_per_second')
            if device_usage.pcie_bandwidth_rx_mbps > 0:
                store.set_gauge(
                    name='gpu.pcie.bandwidth.rx', tags=metric_tags, 
                    value=device_usage.pcie_bandwidth_rx_mbps, update_ts=now, unit='megabits_per_second')
            if device_usage.pcie_max_bandwidth_gbps > 0:
                store.set_gauge(
                    name='gpu.pcie.max_bandwidth', tags=metric_tags, 
                    value=device_usage.pcie_max_bandwidth_gbps, update_ts=now, unit='gigabits_per_second')
            
            # NVLINK metrics
            if device_usage.nvlink_throughput_data_tx_kibs > 0:
                store.set_gauge(
                    name='gpu.nvlink.throughput.data.tx', tags=metric_tags, 
                    value=device_usage.nvlink_throughput_data_tx_kibs, update_ts=now, unit='kibibytes_per_second')
            if device_usage.nvlink_throughput_data_rx_kibs > 0:
                store.set_gauge(
                    name='gpu.nvlink.throughput.data.rx', tags=metric_tags, 
                    value=device_usage.nvlink_throughput_data_rx_kibs, update_ts=now, unit='kibibytes_per_second')
            if device_usage.nvlink_throughput_raw_tx_kibs > 0:
                store.set_gauge(
                    name='gpu.nvlink.throughput.raw.tx', tags=metric_tags, 
                    value=device_usage.nvlink_throughput_raw_tx_kibs, update_ts=now, unit='kibibytes_per_second')
            if device_usage.nvlink_throughput_raw_rx_kibs > 0:
                store.set_gauge(
                    name='gpu.nvlink.throughput.raw.rx', tags=metric_tags, 
                    value=device_usage.nvlink_throughput_raw_rx_kibs, update_ts=now, unit='kibibytes_per_second')
            if device_usage.nvlink_bandwidth_tx_gbps > 0:
                store.set_gauge(
                    name='gpu.nvlink.bandwidth.tx', tags=metric_tags, 
                    value=device_usage.nvlink_bandwidth_tx_gbps, update_ts=now, unit='gigabits_per_second')
            if device_usage.nvlink_bandwidth_rx_gbps > 0:
                store.set_gauge(
                    name='gpu.nvlink.bandwidth.rx', tags=metric_tags, 
                    value=device_usage.nvlink_bandwidth_rx_gbps, update_ts=now, unit='gigabits_per_second')
            if device_usage.nvlink_utilization_tx_percent > 0:
                store.set_gauge(
                    name='gpu.nvlink.utilization.tx', tags=metric_tags, 
                    value=device_usage.nvlink_utilization_tx_percent, update_ts=now, unit='percent')
            if device_usage.nvlink_utilization_rx_percent > 0:
                store.set_gauge(
                    name='gpu.nvlink.utilization.rx', tags=metric_tags, 
                    value=device_usage.nvlink_utilization_rx_percent, update_ts=now, unit='percent')
            if device_usage.nvlink_link_count > 0:
                store.set_gauge(
                    name='gpu.nvlink.link_count', tags=metric_tags, 
                    value=device_usage.nvlink_link_count, update_ts=now)
            if device_usage.nvlink_active_links > 0:
                store.set_gauge(
                    name='gpu.nvlink.active_links', tags=metric_tags, 
                    value=device_usage.nvlink_active_links, update_ts=now)
            if device_usage.nvlink_link_speed_gbps > 0:
                store.set_gauge(
                    name='gpu.nvlink.link_speed', tags=metric_tags, 
                    value=device_usage.nvlink_link_speed_gbps, update_ts=now, unit='gigabits_per_second')
            if device_usage.nvlink_link_width > 0:
                store.set_gauge(
                    name='gpu.nvlink.link_width', tags=metric_tags, 
                    value=device_usage.nvlink_link_width, update_ts=now)
            
            # Error metrics
            if device_usage.ecc_sbe_volatile_total > 0:
                store.set_gauge(
                    name='gpu.errors.ecc.sbe.volatile', tags=metric_tags,
                    value=device_usage.ecc_sbe_volatile_total, update_ts=now)
            if device_usage.ecc_dbe_volatile_total > 0:
                store.set_gauge(
                    name='gpu.errors.ecc.dbe.volatile', tags=metric_tags,
                    value=device_usage.ecc_dbe_volatile_total, update_ts=now)
            if device_usage.ecc_sbe_aggregate_total > 0:
                store.set_gauge(
                    name='gpu.errors.ecc.sbe.aggregate', tags=metric_tags,
                    value=device_usage.ecc_sbe_aggregate_total, update_ts=now)
            if device_usage.ecc_dbe_aggregate_total > 0:
                store.set_gauge(
                    name='gpu.errors.ecc.dbe.aggregate', tags=metric_tags,
                    value=device_usage.ecc_dbe_aggregate_total, update_ts=now)
            if device_usage.pcie_replay_counter > 0:
                store.set_gauge(
                    name='gpu.errors.pcie.replay', tags=metric_tags,
                    value=device_usage.pcie_replay_counter, update_ts=now)
            if device_usage.nvlink_replay_errors > 0:
                store.set_gauge(
                    name='gpu.errors.nvlink.replay', tags=metric_tags,
                    value=device_usage.nvlink_replay_errors, update_ts=now)
            if device_usage.nvlink_recovery_errors > 0:
                store.set_gauge(
                    name='gpu.errors.nvlink.recovery', tags=metric_tags,
                    value=device_usage.nvlink_recovery_errors, update_ts=now)
            if device_usage.nvlink_crc_errors > 0:
                store.set_gauge(
                    name='gpu.errors.nvlink.crc', tags=metric_tags,
                    value=device_usage.nvlink_crc_errors, update_ts=now)
            if device_usage.nvlink_minor_errors > 0:
                store.set_gauge(
                    name='gpu.errors.nvlink.minor', tags=metric_tags,
                    value=device_usage.nvlink_minor_errors, update_ts=now)
            if device_usage.nvlink_major_errors > 0:
                store.set_gauge(
                    name='gpu.errors.nvlink.major', tags=metric_tags,
                    value=device_usage.nvlink_major_errors, update_ts=now)
            if device_usage.nvlink_fatal_errors > 0:
                store.set_gauge(
                    name='gpu.errors.nvlink.fatal', tags=metric_tags,
                    value=device_usage.nvlink_fatal_errors, update_ts=now)
            if device_usage.retired_pages_sbe > 0:
                store.set_gauge(
                    name='gpu.errors.retired_pages.sbe', tags=metric_tags,
                    value=device_usage.retired_pages_sbe, update_ts=now)
            if device_usage.retired_pages_dbe > 0:
                store.set_gauge(
                    name='gpu.errors.retired_pages.dbe', tags=metric_tags,
                    value=device_usage.retired_pages_dbe, update_ts=now)

            # XID errors
            num_xid_errors = len(device_usage.last_xid_error_codes)
            if num_xid_errors > 0:
                store.inc_counter(
                    name='gpu.errors.xid', tags=metric_tags, 
                    value=num_xid_errors, update_ts=now)
                for xid_error_code in device_usage.last_xid_error_codes:
                    graphsignal._tracer.report_error(
                        name='gpu.errors.xid',
                        tags=metric_tags,
                        level='error',
                        message=f'XID error {xid_error_code}'
                    )
                device_usage.last_xid_error_codes = []

    def take_snapshot(self) -> List[DeviceUsage]:
        if not self._is_initialized:
            return []

        current_pid = os.getpid()

        device_usages: List[DeviceUsage] = []

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
            except Exception as err:
                _log_nvml_error(err)

            try:
                handle = nvmlDeviceGetHandleByIndex(idx)
            except Exception as err:
                _log_nvml_error(err)
                continue

            device_usage.device_idx = idx

            try:
                device_usage.device_uuid = nvmlDeviceGetUUID(handle)
            except Exception as err:
                _log_nvml_error(err)

            try:
                device_usage.device_name = nvmlDeviceGetName(handle)
            except Exception as err:
                _log_nvml_error(err)

            try:
                pci_info = nvmlDeviceGetPciInfo_v3(handle)
                device_usage.bus_id = pci_info.busId
            except Exception as err:
                # Fallback to v1 API
                pci_info = nvmlDeviceGetPciInfo(handle)
                device_usage.bus_id = pci_info.busId

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
                elif arch == NVML_DEVICE_ARCH_BLACKWELL:
                    device_usage.architecture = 'Blackwell'
                else:
                    device_usage.architecture = f'Unknown({arch})'
            except Exception as err:
                _log_nvml_error(err)

            try:
                cc_major, cc_minor = nvmlDeviceGetCudaComputeCapability(handle)
                device_usage.compute_capability = SemVer()
                device_usage.compute_capability.major = cc_major
                device_usage.compute_capability.minor = cc_minor
            except Exception as err:
                _log_nvml_error(err)

            try:
                try:
                    mem_info = nvmlDeviceGetMemoryInfo_v2(handle)
                    device_usage.mem_total = mem_info.total
                    device_usage.mem_used = mem_info.used
                    device_usage.mem_free = mem_info.free
                    device_usage.mem_reserved = mem_info.reserved
                except Exception as err:
                    mem_info = nvmlDeviceGetMemoryInfo(handle)
                    device_usage.mem_total = mem_info.total
                    device_usage.mem_used = mem_info.used
                    device_usage.mem_free = mem_info.free
                    device_usage.mem_reserved = 0  # Not available in v1
            except Exception as err:
                _log_nvml_error(err)

            seen_pids = set()
            process_info_fns = [
                nvmlDeviceGetComputeRunningProcesses_v3,  # Use v2 API first
                nvmlDeviceGetComputeRunningProcesses,     # Fallback to v1
                nvmlDeviceGetMPSComputeRunningProcesses_v3,  # Use v2 API first
                nvmlDeviceGetMPSComputeRunningProcesses,     # Fallback to v1
                nvmlDeviceGetGraphicsRunningProcesses_v3,    # Use v2 API first
                nvmlDeviceGetGraphicsRunningProcesses]       # Fallback to v1
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
                            if hasattr(process_info, 'usedGpuMemory') and process_info.usedGpuMemory:
                                device_process_usage.mem_used = process_info.usedGpuMemory
                except Exception as err:
                    _log_nvml_error(err)

            try:
                last_read_us = max(
                    int(self._setup_us or 0),
                    now_us - NVMLRecorder.MIN_SAMPLE_READ_INTERVAL_US)

                sample_value_type, gpu_samples = nvmlDeviceGetSamples(handle, NVML_GPU_UTILIZATION_SAMPLES, last_read_us)
                device_usage.gpu_utilization_percent = _avg_sample_value(sample_value_type, gpu_samples)

                sample_value_type, mem_samples = nvmlDeviceGetSamples(handle, NVML_MEMORY_UTILIZATION_SAMPLES, last_read_us)
                device_usage.mem_access_percent = _avg_sample_value(sample_value_type, mem_samples)
            except Exception as err:
                _log_nvml_error(err)

            try:
                device_usage.pcie_throughput_tx = nvmlDeviceGetPcieThroughput(
                    handle, NVML_PCIE_UTIL_TX_BYTES)
                device_usage.pcie_throughput_rx = nvmlDeviceGetPcieThroughput(
                    handle, NVML_PCIE_UTIL_RX_BYTES)
                
                # Calculate PCIe bandwidth in Mbps
                device_usage.pcie_bandwidth_tx_mbps = device_usage.pcie_throughput_tx * 8 / 1e6
                device_usage.pcie_bandwidth_rx_mbps = device_usage.pcie_throughput_rx * 8 / 1e6
                
                # Get PCIe generation and width
                try:
                    pci_info = nvmlDeviceGetPciInfo_v3(handle)
                    # v3 API provides additional PCIe information
                    if hasattr(pci_info, 'pcieGen'):
                        device_usage.pcie_gen = pci_info.pcieGen
                    if hasattr(pci_info, 'pcieWidth'):
                        device_usage.pcie_width = pci_info.pcieWidth
                    if hasattr(pci_info, 'pcieMaxLinkGen'):
                        device_usage.pcie_max_gen = pci_info.pcieMaxLinkGen
                    if hasattr(pci_info, 'pcieMaxLinkWidth'):
                        device_usage.pcie_max_width = pci_info.pcieMaxLinkWidth

                    # Calculate max theoretical bandwidth based on gen and width
                    # PCIe bandwidth calculation: gen_speed * width * encoding_overhead
                    gen_speeds = {1: 2.5, 2: 5.0, 3: 8.0, 4: 16.0, 5: 32.0, 6: 64.0}  # GT/s per lane
                    if device_usage.pcie_gen is not None and device_usage.pcie_width is not None and device_usage.pcie_gen in gen_speeds and device_usage.pcie_width > 0:
                        # 8b/10b encoding for Gen1-2, 128b/130b for Gen3+
                        encoding_overhead = 0.8 if device_usage.pcie_gen <= 2 else 0.9846
                        device_usage.pcie_max_bandwidth_gbps = (
                            gen_speeds[device_usage.pcie_gen] * 
                            device_usage.pcie_width * 
                            encoding_overhead
                        )
                except Exception as err:
                    _log_nvml_error(err)
                
                # Calculate PCIe utilization percentages
                if device_usage.pcie_max_bandwidth_gbps > 0:
                    device_usage.pcie_utilization_tx_percent = (device_usage.pcie_bandwidth_tx_mbps / (device_usage.pcie_max_bandwidth_gbps * 1000)) * 100
                    device_usage.pcie_utilization_rx_percent = (device_usage.pcie_bandwidth_rx_mbps / (device_usage.pcie_max_bandwidth_gbps * 1000)) * 100
                    
            except Exception as err:
                _log_nvml_error(err)

            try:
                nvlink_throughput_data_tx = nvmlDeviceGetFieldValues(handle, [NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_TX])[0]
                if nvlink_throughput_data_tx.nvmlReturn == NVML_SUCCESS:
                    value = _nvml_value(nvlink_throughput_data_tx.valueType, nvlink_throughput_data_tx.value)
                    if idx in self._last_nvlink_throughput_data_tx:
                        last_data = self._last_nvlink_throughput_data_tx[idx]
                        last_value = _nvml_value(last_data.valueType, last_data.value)
                        if last_value is not None and value is not None:
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
                        if last_value is not None and value is not None:
                            interval_us = nvlink_throughput_data_rx.timestamp - last_data.timestamp
                            if interval_us > 0:
                                device_usage.nvlink_throughput_data_rx_kibs = (value - last_value) / (interval_us * 1e6)
                    self._last_nvlink_throughput_data_rx[idx] = nvlink_throughput_data_rx

                # NVLINK raw throughput
                try:
                    nvlink_throughput_raw_tx = nvmlDeviceGetFieldValues(handle, [NVML_FI_DEV_NVLINK_THROUGHPUT_RAW_TX])[0]
                    if nvlink_throughput_raw_tx.nvmlReturn == NVML_SUCCESS:
                        value = _nvml_value(nvlink_throughput_raw_tx.valueType, nvlink_throughput_raw_tx.value)
                        if idx in self._last_nvlink_throughput_raw_tx:
                            last_data = self._last_nvlink_throughput_raw_tx[idx]
                            last_value = _nvml_value(last_data.valueType, last_data.value)
                            if last_value is not None and value is not None:
                                interval_us = nvlink_throughput_raw_tx.timestamp - last_data.timestamp
                                if interval_us > 0:
                                    device_usage.nvlink_throughput_raw_tx_kibs = (value - last_value) / (interval_us * 1e6)
                        self._last_nvlink_throughput_raw_tx[idx] = nvlink_throughput_raw_tx
                except Exception as err:
                    _log_nvml_error(err)

                try:
                    nvlink_throughput_raw_rx = nvmlDeviceGetFieldValues(handle, [NVML_FI_DEV_NVLINK_THROUGHPUT_RAW_RX])[0]
                    if nvlink_throughput_raw_rx.nvmlReturn == NVML_SUCCESS:
                        value = _nvml_value(nvlink_throughput_raw_rx.valueType, nvlink_throughput_raw_rx.value)
                        if idx in self._last_nvlink_throughput_raw_rx:
                            last_data = self._last_nvlink_throughput_raw_rx[idx]
                            last_value = _nvml_value(last_data.valueType, last_data.value)
                            if last_value is not None and value is not None:
                                interval_us = nvlink_throughput_raw_rx.timestamp - last_data.timestamp
                                if interval_us > 0:
                                    device_usage.nvlink_throughput_raw_rx_kibs = (value - last_value) / (interval_us * 1e6)
                        self._last_nvlink_throughput_raw_rx[idx] = nvlink_throughput_raw_rx
                except Exception as err:
                    _log_nvml_error(err)

                # Calculate NVLINK bandwidth in Gbps
                device_usage.nvlink_bandwidth_tx_gbps = (device_usage.nvlink_throughput_data_tx_kibs + device_usage.nvlink_throughput_raw_tx_kibs) * 1024 * 8 / 1e9
                device_usage.nvlink_bandwidth_rx_gbps = (device_usage.nvlink_throughput_data_rx_kibs + device_usage.nvlink_throughput_raw_rx_kibs) * 1024 * 8 / 1e9
                
            except Exception as err:
                _log_nvml_error(err)

            try:
                device_usage.gpu_temp_c = nvmlDeviceGetTemperature(
                    handle, NVML_TEMPERATURE_GPU)
            except Exception as err:
                _log_nvml_error(err)

            try:
                device_usage.power_usage_w = nvmlDeviceGetPowerUsage(handle) / 1000.0
            except Exception as err:
                _log_nvml_error(err)

            try:
                device_usage.fan_speed_percent = nvmlDeviceGetFanSpeed(handle)
            except Exception as err:
                _log_nvml_error(err)

            try:
                device_usage.ecc_sbe_volatile_total = nvmlDeviceGetTotalEccErrors(
                    handle, NVML_MEMORY_ERROR_TYPE_CORRECTED, NVML_VOLATILE_ECC)
            except Exception as err:
                _log_nvml_error(err)

            try:
                device_usage.ecc_dbe_volatile_total = nvmlDeviceGetTotalEccErrors(
                    handle, NVML_MEMORY_ERROR_TYPE_UNCORRECTED, NVML_VOLATILE_ECC)
            except Exception as err:
                _log_nvml_error(err)

            try:
                device_usage.ecc_sbe_aggregate_total = nvmlDeviceGetTotalEccErrors(
                    handle, NVML_MEMORY_ERROR_TYPE_CORRECTED, NVML_AGGREGATE_ECC)
            except Exception as err:
                _log_nvml_error(err)

            try:
                device_usage.ecc_dbe_aggregate_total = nvmlDeviceGetTotalEccErrors(
                    handle, NVML_MEMORY_ERROR_TYPE_UNCORRECTED, NVML_AGGREGATE_ECC)
            except Exception as err:
                _log_nvml_error(err)

            try:
                device_usage.pcie_replay_counter = nvmlDeviceGetPcieReplayCounter(handle)
            except Exception as err:
                _log_nvml_error(err)

            try:
                # Check if device supports NVLINK by trying to get NVLINK count
                nvlink_count = 0
                active_links = 0
                total_link_speed = 0.0
                total_link_width = 0
                
                for link in range(6):  # Maximum 6 NVLINK connections
                    try:
                        state = nvmlDeviceGetNvLinkState(handle, link)
                        if state:
                            nvlink_count += 1
                            if state:  # Link is active
                                active_links += 1
                                
                                # Get link speed and width
                                try:
                                    link_speed = nvmlDeviceGetNvLinkSpeed(handle, link)
                                    total_link_speed += link_speed
                                except Exception:
                                    pass
                                
                                try:
                                    link_width = nvmlDeviceGetNvLinkWidth(handle, link)
                                    total_link_width += link_width
                                except Exception:
                                    pass
                    except Exception:
                        break
                
                device_usage.nvlink_link_count = nvlink_count
                device_usage.nvlink_active_links = active_links
                if active_links > 0:
                    device_usage.nvlink_link_speed_gbps = total_link_speed / active_links
                    device_usage.nvlink_link_width = int(total_link_width / active_links)
                
                if nvlink_count > 0:
                    for link in range(nvlink_count):
                        try:
                            device_usage.nvlink_replay_errors += nvmlDeviceGetNvLinkErrorCounter(
                                handle, link, NVML_NVLINK_ERROR_DL_REPLAY)
                        except Exception as err:
                            _log_nvml_error(err)
                        
                        try:
                            device_usage.nvlink_recovery_errors += nvmlDeviceGetNvLinkErrorCounter(
                                handle, link, NVML_NVLINK_ERROR_DL_RECOVERY)
                        except Exception as err:
                            _log_nvml_error(err)
                        
                        try:
                            device_usage.nvlink_crc_errors += nvmlDeviceGetNvLinkErrorCounter(
                                handle, link, NVML_NVLINK_ERROR_DL_CRC)
                        except Exception as err:
                            _log_nvml_error(err)
                        
                        try:
                            device_usage.nvlink_minor_errors += nvmlDeviceGetNvLinkErrorCounter(
                                handle, link, NVML_NVLINK_ERROR_DL_MINOR)
                        except Exception as err:
                            _log_nvml_error(err)
                        
                        try:
                            device_usage.nvlink_major_errors += nvmlDeviceGetNvLinkErrorCounter(
                                handle, link, NVML_NVLINK_ERROR_DL_MAJOR)
                        except Exception as err:
                            _log_nvml_error(err)
                        
                        try:
                            device_usage.nvlink_fatal_errors += nvmlDeviceGetNvLinkErrorCounter(
                                handle, link, NVML_NVLINK_ERROR_DL_FATAL)
                        except Exception as err:
                            _log_nvml_error(err)
                
                # Calculate NVLINK utilization percentages
                # Theoretical max NVLINK bandwidth varies by generation (NVLink 1.0: 20GB/s, NVLink 2.0: 25GB/s, NVLink 3.0: 50GB/s, NVLink 4.0: 100GB/s)
                # For simplicity, we'll use a conservative estimate based on link count and speed
                if device_usage.nvlink_link_speed_gbps > 0 and device_usage.nvlink_link_width > 0:
                    max_theoretical_bandwidth_gbps = device_usage.nvlink_link_speed_gbps * device_usage.nvlink_link_width * active_links
                    if max_theoretical_bandwidth_gbps > 0:
                        device_usage.nvlink_utilization_tx_percent = (device_usage.nvlink_bandwidth_tx_gbps / max_theoretical_bandwidth_gbps) * 100
                        device_usage.nvlink_utilization_rx_percent = (device_usage.nvlink_bandwidth_rx_gbps / max_theoretical_bandwidth_gbps) * 100
                        
            except Exception as err:
                # NVLINK not available on this device
                pass

            # Retired pages
            '''
            try:
                try:
                    retired_pages_sbe = nvmlDeviceGetRetiredPages_v2(
                        handle, NVML_PAGE_RETIREMENT_CAUSE_MULTIPLE_SINGLE_BIT_ECC_ERRORS)
                    device_usage.retired_pages_sbe = len(retired_pages_sbe) if retired_pages_sbe else 0
                except Exception as err:
                    retired_pages_sbe = nvmlDeviceGetRetiredPages(
                        handle, NVML_PAGE_RETIREMENT_CAUSE_MULTIPLE_SINGLE_BIT_ECC_ERRORS)
                    device_usage.retired_pages_sbe = len(retired_pages_sbe) if retired_pages_sbe else 0
            except Exception as err:
                _log_nvml_error(err)

            try:
                try:
                    retired_pages_dbe = nvmlDeviceGetRetiredPages_v2(
                        handle, NVML_PAGE_RETIREMENT_CAUSE_DOUBLE_BIT_ECC_ERROR)
                    device_usage.retired_pages_dbe = len(retired_pages_dbe) if retired_pages_dbe else 0
                except Exception as err:
                    retired_pages_dbe = nvmlDeviceGetRetiredPages(
                        handle, NVML_PAGE_RETIREMENT_CAUSE_DOUBLE_BIT_ECC_ERROR)
                    device_usage.retired_pages_dbe = len(retired_pages_dbe) if retired_pages_dbe else 0
            except Exception as err:
                _log_nvml_error(err)
            '''

            # Check for MIG (Multi-Instance GPU) support
            try:
                mig_mode = nvmlDeviceGetMigMode(handle)
                if mig_mode[0]:  # MIG mode is enabled
                    device_usage.device_type = 'mig'
                    # Get MIG device info if available
                    try:
                        mig_device_count = nvmlDeviceGetMaxMigDeviceCount(handle)
                        if mig_device_count > 0:
                            logger.debug(f'Device {idx} supports {mig_device_count} MIG instances')
                    except Exception:
                        pass
                else:
                    device_usage.device_type = 'gpu'
            except Exception:
                device_usage.device_type = 'gpu'  # Default to GPU if MIG check fails

            # Update XID error information from error counters
            if idx in self._error_counters:
                device_usage.last_xid_error_codes = self._error_counters[idx]['last_xid_error_codes']


        self._last_snapshot = device_usages
        return device_usages


def _avg_sample_value(sample_value_type, samples):
    if not samples:
        return 0.0

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

    return 0.0


def _nvml_value(value_type, value) -> Optional[Union[int, float]]:
    if value_type == NVML_VALUE_TYPE_DOUBLE:
        return value.dVal
    if value_type == NVML_VALUE_TYPE_UNSIGNED_INT:
        return value.uiVal
    if value_type == NVML_VALUE_TYPE_UNSIGNED_LONG:
        return value.ulVal
    if value_type == NVML_VALUE_TYPE_UNSIGNED_LONG_LONG:
        return value.ullVal
    return None


def _log_nvml_error(err):
    if hasattr(err, 'value'):
        if (err.value == NVML_ERROR_NOT_FOUND):
            pass
        elif (err.value == NVML_ERROR_NOT_SUPPORTED):
            logger.debug('NVML call not supported', exc_info=True)
        elif (err.value == NVML_ERROR_INVALID_ARGUMENT):
            logger.debug(f'NVML call invalid argument', exc_info=True)
        else:
            logger.error('Error calling NVML', exc_info=True)
    else:
        logger.error('Exception calling NVML', exc_info=True)


def _format_version(version):
    major = int(version / 1000)
    minor = int(version % 1000 / 10)
    return '{0}.{1}'.format(major, minor)
