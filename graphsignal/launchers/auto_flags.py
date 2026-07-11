"""`graphsignal-run --auto-flags` support.

`inject_auto_flags()` runs inside the launcher process (before `execv` and
before the SDK exists), POSTs the run context (tags, engine command line,
system attributes) to the Graphsignal `/api/v1/auto_flags` endpoint, and
merges the recommended engine flags returned by the server into the engine
command line. It is strictly best-effort: any failure (missing API key,
network error, malformed response) leaves the caller's args unchanged so the
workload always launches.
"""

import json
import logging
import os
import platform
import shlex
import socket
from typing import Any, Dict, List, Optional

import requests

from graphsignal.sdk.env_vars import read_config_tags

logger = logging.getLogger('graphsignal')

DEFAULT_API_BASE = 'https://api.graphsignal.com'

_REQUEST_TIMEOUT = (5, 10)


def inject_auto_flags(args: List[str], engine_name: Optional[str] = None,
                      engine_version: Optional[str] = None) -> List[str]:
    """Best-effort. On any failure, returns args unchanged."""
    try:
        api_key = os.getenv('GRAPHSIGNAL_API_KEY')
        if not api_key:
            logger.debug('auto-flags: GRAPHSIGNAL_API_KEY not set; skipping')
            return args

        api_base = os.getenv('GRAPHSIGNAL_API_BASE') or DEFAULT_API_BASE

        body = _build_request_body(args, engine_name, engine_version)

        options = _fetch_options(api_base, api_key, body)
        if not options:
            return args

        return _merge_options(args, options)
    except Exception:
        logger.debug('auto-flags: error injecting auto flags', exc_info=True)
        return args


def _build_request_body(args: List[str], engine_name: Optional[str],
                        engine_version: Optional[str]) -> Dict[str, Any]:
    tags = read_config_tags()
    try:
        hostname = socket.gethostname()
        if hostname:
            tags['host.name'] = hostname
    except Exception:
        logger.debug('auto-flags: error reading hostname', exc_info=True)

    return {
        'tags': tags,
        'command_line': shlex.join(args),
        'system': _collect_system(engine_name, engine_version),
    }


def _collect_system(engine_name: Optional[str],
                    engine_version: Optional[str]) -> List[Dict[str, Any]]:
    system: List[Dict[str, Any]] = []

    def add(name: str, value: Any) -> None:
        if value is not None and value != '':
            system.append({'name': name, 'value': value})

    try:
        add('platform.name', platform.system())
        add('platform.version', platform.release())
        add('platform.machine', platform.machine())
    except Exception:
        logger.debug('auto-flags: error reading platform info', exc_info=True)

    _collect_device_attrs(add)

    add('engine.name', engine_name)
    add('engine.version', engine_version)

    return system


# NVML architecture id -> human-readable name (mirrors nvml_recorder.py).
_NVML_ARCH_NAMES = {
    2: 'Kepler',
    3: 'Maxwell',
    4: 'Pascal',
    5: 'Volta',
    6: 'Turing',
    7: 'Ampere',
    8: 'Ada',
    9: 'Hopper',
    10: 'Blackwell',
}


def _collect_device_attrs(add) -> None:
    try:
        import pynvml
    except Exception:
        logger.debug('auto-flags: pynvml not available; skipping device info',
                     exc_info=True)
        return

    initialized = False
    try:
        pynvml.nvmlInit()
        initialized = True

        device_count = pynvml.nvmlDeviceGetCount()
        if device_count <= 0:
            add('device.count', 0)
            return

        visible_idxs = _visible_device_idxs(device_count)
        add('device.count', len(visible_idxs))

        # `i` is the position among visible devices (device.0, device.1, ...);
        # `nvml_idx` is the underlying NVML device index.
        for i, nvml_idx in enumerate(visible_idxs):
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(nvml_idx)
            except Exception:
                logger.debug('auto-flags: error getting handle for device %s',
                             nvml_idx, exc_info=True)
                continue
            _collect_single_device_attrs(add, pynvml, handle, i)
    except Exception:
        logger.debug('auto-flags: error collecting device info', exc_info=True)
    finally:
        if initialized:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass


def _visible_device_idxs(device_count: int) -> List[int]:
    """Resolve visible NVML device indices, honoring CUDA_VISIBLE_DEVICES."""
    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    if cuda_visible_devices:
        try:
            return [int(x) for x in cuda_visible_devices.split(',') if x.strip() != '']
        except Exception:
            logger.debug('auto-flags: could not parse CUDA_VISIBLE_DEVICES=%r',
                         cuda_visible_devices, exc_info=True)
    return list(range(device_count))


def _collect_single_device_attrs(add, pynvml, handle, i: int) -> None:
    prefix = f'device.{i}'

    try:
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode('utf-8', 'replace')
        add(f'{prefix}.device_name', name)
    except Exception:
        logger.debug('auto-flags: error reading device name', exc_info=True)

    try:
        arch = pynvml.nvmlDeviceGetArchitecture(handle)
        add(f'{prefix}.architecture', _NVML_ARCH_NAMES.get(arch, f'Unknown({arch})'))
    except Exception:
        logger.debug('auto-flags: error reading device architecture', exc_info=True)

    try:
        cc_major, cc_minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
        add(f'{prefix}.compute_capability', f'{cc_major}.{cc_minor}')
    except Exception:
        logger.debug('auto-flags: error reading compute capability', exc_info=True)

    try:
        try:
            mem_info = pynvml.nvmlDeviceGetMemoryInfo_v2(handle)
        except Exception:
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        add(f'{prefix}.mem_total', mem_info.total)
    except Exception:
        logger.debug('auto-flags: error reading device memory', exc_info=True)


def _fetch_options(api_base: str, api_key: str,
                   body: Dict[str, Any]) -> List[Dict[str, Any]]:
    url = f"{api_base}/api/v1/auto_flags"
    headers = {'X-API-Key': api_key}

    logger.debug('auto-flags: requesting %s', url)
    logger.debug('auto-flags: payload %s', json.dumps(body, indent=2))
    resp = requests.post(url, json=body, headers=headers, timeout=_REQUEST_TIMEOUT)
    resp.raise_for_status()

    options = resp.json()
    if not isinstance(options, list):
        logger.debug('auto-flags: unexpected response type: %s', type(options))
        return []
    return options


def _merge_options(args: List[str],
                   options: List[Dict[str, Any]]) -> List[str]:
    merged = list(args)
    for option in options:
        if not isinstance(option, dict):
            continue
        for flag_str in option.get('args', []) or []:
            tokens = shlex.split(flag_str)
            if not tokens:
                continue
            flag = tokens[0]
            merged = _apply_flag(merged, flag, tokens)
    return merged


def _apply_flag(args: List[str], flag: str, tokens: List[str]) -> List[str]:
    """Append tokens, or, when `flag` already exists, replace its occurrence."""
    idx = _find_flag(args, flag)
    if idx is None:
        logger.info('auto-flags: adding %s', ' '.join(tokens))
        return args + tokens

    logger.info('auto-flags: modifying %s', ' '.join(tokens))
    new_args = list(args)
    existing = new_args[idx]
    if '=' in existing and existing.startswith(flag + '='):
        # `--flag=value` form occupies a single argv entry.
        new_args[idx:idx + 1] = tokens
    else:
        # `--flag value` form: drop the flag and, if present, its value.
        end = idx + 1
        if end < len(new_args) and not new_args[end].startswith('-'):
            end += 1
        new_args[idx:end] = tokens
    return new_args


def _find_flag(args: List[str], flag: str) -> Optional[int]:
    for i, arg in enumerate(args):
        if arg == flag or arg.startswith(flag + '='):
            return i
    return None
