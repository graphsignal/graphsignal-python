import glob
import hashlib
import json
import logging
import os
import re
import shutil
import sys
import threading
import time
from typing import Any, Dict, List, Optional

import graphsignal
import graphsignal.sdk
from graphsignal.recorders.base_recorder import BaseRecorder

logger = logging.getLogger('graphsignal')

TOP_N_KERNELS = 30
DEFAULT_RESOLUTION_NS = 10_000_000        # 10 ms
DEFAULT_ACTIVITY_WINDOW_NS = 1_000_000_000  # 1 s


class _EventFields:
    __slots__ = ['cumtime_field_id', 'ncalls_field_id', 'nerrors_field_id',
                 'bytes_field_id', 'cumtime_occupancy_field_id',
                 'host_sync_wait_field_id']

    def __init__(self, cumtime_field_id=None, ncalls_field_id=None, nerrors_field_id=None,
                 bytes_field_id=None, cumtime_occupancy_field_id=None,
                 host_sync_wait_field_id=None):
        self.cumtime_field_id = cumtime_field_id
        self.ncalls_field_id = ncalls_field_id
        self.nerrors_field_id = nerrors_field_id
        self.bytes_field_id = bytes_field_id
        self.cumtime_occupancy_field_id = cumtime_occupancy_field_id
        self.host_sync_wait_field_id = host_sync_wait_field_id


# ----- Op-name extraction pipeline -----------------------------------------
# Derives a short, displayable canonical op name from a raw CUPTI kernel
# symbol (Itanium-mangled-C++ unwrap, CUDA arch-prefix strip, library suffix
# strip, Triton seq-number strip). Kernel *categorization* lives in the
# platform; this is purely a name-*shape* transform, kept here so op_names are
# stable at emit time. `make_op_name` appends a short hex fingerprint of the
# raw symbol so kernels that collapse to the same family stay distinguishable.

MAX_OP_NAME_LEN = 60

# Generic-boilerplate identifiers that, when seen as the *last* element of a
# mangled-name identifier chain, get prefixed with the previous identifier so
# the function still has its surrounding context.
_BOILERPLATE_LAST_IDENT = frozenset(('kernel', 'Kernel', 'Kernel2', 'Kernel3'))

# Leading CUDA arch / vendor prefix to strip (lowercase NVIDIA conventions).
_ARCH_PREFIX_RE = re.compile(
    r'^(?:sm\d+|sm_\d+|volta|turing|ampere|ada|hopper|blackwell)_')

# Embedded `_sm<digits>_` in the middle of a name (e.g. `nvjet_sm121_…`).
_MID_SM_RE = re.compile(r'_sm\d+_')

# Trailing CUDA-library suffix flags. Applied repeatedly until stable.
_SUFFIX_STRIP_PATTERNS = (
    re.compile(r'_\d+x\d+(?:x\d+)?$'),                       # _128x64, _128x208x64
    re.compile(r'_[TtNn]{2,5}$'),                            # transpose flags: tn, nt, nn, tt, TNNN
    re.compile(r'_tma[A-Za-z]*$'),                           # nvjet tma block-shape flags
    re.compile(r'_bz$'),                                     # nvjet block-z flag
    re.compile(r'_align\d+$', re.IGNORECASE),                # alignment
    re.compile(r'_[fbi]\d+(?:[fbi]\d+)+$', re.IGNORECASE),   # _f16f16_f32
    re.compile(r'_[fbi]\d+$', re.IGNORECASE),                # _f16
    re.compile(r'_\d+(?:d\d*)*$'),                           # Triton seq: _0, _0d1d2d, _1d2d
)

_OP_NAME_FINGERPRINT_LEN = 4
_OP_NAME_FINGERPRINT_SEP = '@'


def _itanium_identifiers(mangled: str) -> List[str]:
    """Walk an Itanium-mangled C++ symbol `_ZN<len><name><len><name>...E` and
    collect the identifier sequence. Best-effort: stops at the first
    non-numeric byte (template-arg or parameter section). Returns the
    identifiers in order; returns ``[]`` on any unexpected shape."""
    if not mangled.startswith('_Z'):
        return []
    i = 2
    if i < len(mangled) and mangled[i] == 'N':
        i += 1
    idents: List[str] = []
    while i < len(mangled):
        # Read a length prefix (one or more decimal digits).
        j = i
        while j < len(mangled) and mangled[j].isdigit():
            j += 1
        if j == i:
            break  # no length → end of identifier chain
        try:
            length = int(mangled[i:j])
        except ValueError:
            break
        i = j
        if length <= 0 or i + length > len(mangled):
            break
        ident = mangled[i:i + length]
        # Must look like a C++ identifier — alphanumeric / underscore.
        if not all(c.isalnum() or c == '_' for c in ident):
            break
        idents.append(ident)
        i += length
    return idents


def _smart_last_identifier(idents: List[str]) -> Optional[str]:
    """Pick the most-informative trailing identifier. If the last element is
    a generic boilerplate name (``kernel``, ``Kernel2``, …), combine it
    with the previous identifier so the function still has context."""
    if not idents:
        return None
    last = idents[-1]
    if last in _BOILERPLATE_LAST_IDENT and len(idents) >= 2:
        return f'{idents[-2]}_{last}'
    return last


def extract_op_name(kernel_name: str) -> str:
    """Return a short, displayable canonical op name derived from the raw
    CUPTI kernel name. Purely a *shape* transform. Fallback is the raw name
    truncated to ``MAX_OP_NAME_LEN``."""
    if not kernel_name:
        return ''

    name = kernel_name

    if name.startswith('_Z'):
        ident = _smart_last_identifier(_itanium_identifiers(name))
        if ident:
            name = ident

    name = _ARCH_PREFIX_RE.sub('', name)
    name = _MID_SM_RE.sub('_', name, count=1)

    while True:
        prev = name
        for pat in _SUFFIX_STRIP_PATTERNS:
            name = pat.sub('', name)
        if name == prev:
            break

    if not name:
        name = kernel_name
    if len(name) > MAX_OP_NAME_LEN:
        name = name[:MAX_OP_NAME_LEN]
    return name


def _short_fingerprint(kernel_name: str) -> str:
    return hashlib.md5(kernel_name.encode('utf-8')).hexdigest()[:_OP_NAME_FINGERPRINT_LEN]


def make_op_name(kernel_name: str) -> str:
    """Return the descriptor ``op_name`` for a raw CUPTI kernel symbol:
    ``<family>@<4-hex-md5>``. The family is ``extract_op_name(kernel_name)``;
    the fingerprint is a stable short hash of the raw symbol so that two raw
    kernels which collapse to the same family stay distinguishable as separate
    operations on the platform."""
    if not kernel_name:
        return ''
    family = extract_op_name(kernel_name)
    fp = _short_fingerprint(kernel_name)
    return f'{family}{_OP_NAME_FINGERPRINT_SEP}{fp}'


class CuptiRecorder(BaseRecorder):
    def __init__(self, pid=None, args=None):
        super().__init__(pid=pid, args=args)
        self._disabled = True

        self._resolution_ns: int = DEFAULT_RESOLUTION_NS
        self._activity_window_ns: int = DEFAULT_ACTIVITY_WINDOW_NS

        self._fields: Dict[int, _EventFields] = {}
        self._kernel_cumtime_totals: Dict[int, int] = {}
        self._top_kernel_ids: set = set()
        self._context: Dict[str, str] = {}
        self._last_drain_bucket_ts = 0

        self._drain_stop_event = threading.Event()
        self._drain_thread = None

    def setup(self):
        if not sys.platform.startswith('linux'):
            return
        if self.pid is None:
            logger.debug('CuptiRecorder requires a pid; skipping setup')
            return

        _sweep_stale_shm_dirs()

        self._disabled = False
        self._start_drain_timer()
        logger.debug('CuptiRecorder started for pid=%s', self.pid)

    def shutdown(self):
        if self._disabled:
            return
        try:
            if self._drain_thread:
                self._drain_stop_event.set()
                self._drain_thread.join(timeout=1.0)
                self._drain_thread = None
        finally:
            self._disabled = True
            self._cleanup_shm_dir()

    def set_resolution_ns(self, resolution_ns: int) -> None:
        if resolution_ns < DEFAULT_RESOLUTION_NS:
            resolution_ns = DEFAULT_RESOLUTION_NS
        self._resolution_ns = int(resolution_ns)

    def get_resolution_ns(self) -> int:
        return self._resolution_ns

    def set_activity_window_ns(self, activity_window_ns: int) -> None:
        self._activity_window_ns = int(max(0, activity_window_ns))

    def get_activity_window_ns(self) -> int:
        return self._activity_window_ns

    def _shm_dir(self) -> str:
        return f"/dev/shm/graphsignal_cupti_{self.pid}"

    def _start_drain_timer(self):
        self._last_drain_bucket_ts = 0
        self._drain_stop_event = threading.Event()

        def _drain_loop():
            while not self._drain_stop_event.wait(self._activity_window_ns / 1e9):
                try:
                    result = self._cupti_activity_drain()
                    if result['buckets']:
                        self._convert_to_profile(result['buckets'])
                except Exception as exc:
                    logger.error('Error in CUPTI drain timer: %s', exc, exc_info=True)

        self._drain_thread = threading.Thread(target=_drain_loop, daemon=True)
        self._drain_thread.start()

    def _cupti_activity_drain(self) -> Dict[str, Any]:
        shm_dir = self._shm_dir()
        files = sorted(glob.glob(os.path.join(shm_dir, "cupti_*.json")))
        all_buckets = []
        for filepath in files:
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                all_buckets.extend(data.get('buckets', []))
                ctx = data.get('context')
                if isinstance(ctx, dict):
                    for k, v in ctx.items():
                        if v in (None, ''):
                            continue
                        if k not in self._context:
                            self._context[k] = str(v)
                self._log_workload_entries(data.get('log'))
            except (OSError, json.JSONDecodeError):
                pass

        cutoff = self._last_drain_bucket_ts
        if cutoff == 0:
            cutoff = time.time_ns() - 2 * self._activity_window_ns

        filtered = [b for b in all_buckets if b.get('bucket_ts', 0) > cutoff]
        if filtered:
            self._last_drain_bucket_ts = max(b['bucket_ts'] for b in filtered)

        return {"buckets": filtered}

    def _log_workload_entries(self, entries) -> None:
        if not entries:
            return
        try:
            for entry in entries:
                if isinstance(entry, dict):
                    msg = entry.get('msg', '')
                else:
                    msg = str(entry)
                if not msg:
                    continue
                logger.debug('cupti workload: %s', msg.rstrip())
        except Exception:
            pass

    def _cleanup_shm_dir(self):
        try:
            if os.path.isdir(self._shm_dir()):
                shutil.rmtree(self._shm_dir(), ignore_errors=True)
        except Exception:
            pass

    def _convert_to_profile(self, buckets) -> None:
        for bucket_data in buckets:
            bucket_ts = bucket_data['bucket_ts']

            # Stats (cumtime, ncalls, nerrors, bytes, cumtime_occupancy,
            # host_sync_wait) are computed in the native lib; read them directly.
            computed = []
            for event_id_str, eb in bucket_data['events'].items():
                event_name = eb.get('event_name', '')
                if not event_name:
                    continue
                cumtime = int(eb.get('cumtime', 0) or 0)
                ncalls = int(eb.get('ncalls', 0) or 0)
                if cumtime == 0 and ncalls == 0:
                    continue
                computed.append((event_id_str, event_name, cumtime, eb))

            for event_id_str, event_name, cumtime, _ in computed:
                # Kernels and CUDA graphs compete for the top-N slots; memcpy/
                # memset/sync events are always kept (below) so they never crowd
                # these out.
                if event_name.startswith('kernel:') or event_name.startswith('graph:'):
                    eid = int(event_id_str)
                    self._kernel_cumtime_totals[eid] = (
                        self._kernel_cumtime_totals.get(eid, 0) + cumtime)
            ranked = sorted(self._kernel_cumtime_totals.items(), key=lambda x: -x[1])
            self._top_kernel_ids = {eid for eid, _ in ranked[:TOP_N_KERNELS]}

            selected = [
                row for row in computed
                if row[1].startswith('memcpy_') or row[1].startswith('sync_') or row[1].startswith('memset_') or int(row[0]) in self._top_kernel_ids
            ]

            profile = {}
            for event_id_str, event_name, cumtime, eb in selected:
                event_id = int(event_id_str)
                fields = self._fields.get(event_id)
                if not fields:
                    # Extra descriptor properties that ride on every field
                    # descriptor for this event (e.g. graph_signature).
                    descriptor_extra = {}
                    if event_name.startswith('memcpy_'):
                        category, display_name = 'cuda.memcpy', event_name
                    elif event_name.startswith('sync_'):
                        category, display_name = 'cuda.sync', event_name
                    elif event_name.startswith('memset_'):
                        category, display_name = 'cuda.memset', event_name
                    elif event_name.startswith('graph:'):
                        # The native lib names graph events `graph:<signature>`
                        # where the signature is a cross-process-stable
                        # structural fingerprint of the graph's nodes. Collapse
                        # to a single `cuda_graph@<hash>` op_name and surface the
                        # raw signature as a descriptor property.
                        signature = event_name[len('graph:'):]
                        category = 'cuda.graph'
                        display_name = (
                            f'cuda_graph{_OP_NAME_FINGERPRINT_SEP}'
                            f'{_short_fingerprint(signature)}')
                        descriptor_extra['graph_signature'] = signature
                    elif event_name.startswith('kernel:'):
                        # All kernels carry the flat `cuda.kernel` category; the
                        # platform refines it to `cuda.kernel.<sub>` from the raw
                        # kernel_name at query time.
                        raw = event_name[len('kernel:'):]
                        category, display_name = 'cuda.kernel', make_op_name(raw)
                        descriptor_extra['kernel_name'] = raw
                    else:
                        # The native lib always prefixes event names; an
                        # unrecognized name is unexpected, so skip it rather than
                        # guessing a category.
                        continue

                    def _descriptor(statistic, unit=None):
                        d = dict(category=category, op_name=display_name, statistic=statistic)
                        if unit is not None:
                            d['unit'] = unit
                        d.update(descriptor_extra)
                        return d

                    cumtime_field_id = graphsignal.sdk.sdk().add_counter_profile_field(
                        descriptor=_descriptor('cumtime', unit='ns'))
                    ncalls_field_id = graphsignal.sdk.sdk().add_counter_profile_field(
                        descriptor=_descriptor('ncalls'))
                    nerrors_field_id = graphsignal.sdk.sdk().add_counter_profile_field(
                        descriptor=_descriptor('nerrors'))
                    bytes_field_id = None
                    if category in ('cuda.memcpy', 'cuda.memset'):
                        bytes_field_id = graphsignal.sdk.sdk().add_counter_profile_field(
                            descriptor=_descriptor('bytes', unit='bytes'))
                    cumtime_occupancy_field_id = None
                    if category == 'cuda.kernel':
                        cumtime_occupancy_field_id = graphsignal.sdk.sdk().add_counter_profile_field(
                            descriptor=_descriptor('cumtime_occupancy', unit='ns'))
                    # Sync events don't get host_sync_wait — they're the
                    # wait source, not something the CPU was blocked on.
                    host_sync_wait_field_id = None
                    if category != 'cuda.sync':
                        host_sync_wait_field_id = graphsignal.sdk.sdk().add_counter_profile_field(
                            descriptor=_descriptor('host_sync_wait', unit='ns'))
                    fields = _EventFields(
                        cumtime_field_id=cumtime_field_id,
                        ncalls_field_id=ncalls_field_id,
                        nerrors_field_id=nerrors_field_id,
                        bytes_field_id=bytes_field_id,
                        cumtime_occupancy_field_id=cumtime_occupancy_field_id,
                        host_sync_wait_field_id=host_sync_wait_field_id)
                    self._fields[event_id] = fields

                if fields.cumtime_field_id and cumtime > 0:
                    profile[fields.cumtime_field_id] = cumtime
                ncalls = int(eb.get('ncalls', 0) or 0)
                if fields.ncalls_field_id and ncalls > 0:
                    profile[fields.ncalls_field_id] = ncalls
                nerrors = int(eb.get('nerrors', 0) or 0)
                if fields.nerrors_field_id and nerrors > 0:
                    profile[fields.nerrors_field_id] = nerrors
                if fields.bytes_field_id:
                    bytes_val = int(eb.get('bytes', 0) or 0)
                    if bytes_val > 0:
                        profile[fields.bytes_field_id] = bytes_val
                if fields.host_sync_wait_field_id:
                    host_sync_wait = int(eb.get('host_sync_wait', 0) or 0)
                    if host_sync_wait > 0:
                        profile[fields.host_sync_wait_field_id] = host_sync_wait
                if fields.cumtime_occupancy_field_id:
                    cumtime_occupancy = int(eb.get('cumtime_occupancy', 0) or 0)
                    if cumtime_occupancy > 0:
                        profile[fields.cumtime_occupancy_field_id] = cumtime_occupancy

            if profile:
                graphsignal.sdk.sdk().update_profile(
                    name='profile.cuda', profile=profile, measurement_ts=bucket_ts,
                    tags=self._profile_tags())

    _CONTEXT_TAG_MAP = {
        'rank': 'process.rank',
        'local_rank': 'process.local_rank',
        'master_addr': 'distributed.master_addr',
        'master_port': 'distributed.master_port',
        'slurm_job_id': 'slurm.job_id',
        'slurm_step_id': 'slurm.step_id',
        'slurm_node_id': 'slurm.node_id',
    }

    def _profile_tags(self):
        tags = {}
        if self.pid is not None:
            tags['process.pid'] = str(self.pid)
        for ctx_key, tag_name in self._CONTEXT_TAG_MAP.items():
            val = self._context.get(ctx_key)
            if val is None:
                continue
            tags[tag_name] = val
        return tags or None


def _sweep_stale_shm_dirs():
    """Remove `/dev/shm/graphsignal_cupti_<pid>` directories whose pid is no
    longer running. The native injection lib creates one per workload pid and
    removes it on clean teardown, but crashes / SIGKILL / OOM leave them behind.
    Called at recorder setup so each profiler run starts by reaping prior leaks.
    Also reaps the legacy `graphsignal_<pid>` prefix written by older lib
    versions (the numeric-pid parse keeps `graphsignal_rocm_*` out of reach)."""
    base = '/dev/shm'
    prefixes = ('graphsignal_cupti_', 'graphsignal_')
    if not os.path.isdir(base):
        return
    try:
        entries = os.listdir(base)
    except OSError:
        return
    for name in entries:
        prefix = next((p for p in prefixes if name.startswith(p)), None)
        if prefix is None:
            continue
        try:
            pid = int(name[len(prefix):])
        except ValueError:
            continue
        if pid <= 0:
            # Defense against weird dir names; `os.kill(0, ...)` targets the
            # current process group and `os.kill(-1, ...)` broadcasts. Never
            # treat such a dir as stale.
            continue
        # `os.kill(pid, 0)` raises ProcessLookupError if the pid is gone,
        # PermissionError if it exists but is owned by another user (still
        # a real process — skip), and returns None if the pid is ours.
        try:
            os.kill(pid, 0)
            continue  # pid alive — leave its dir alone
        except ProcessLookupError:
            pass  # stale — fall through to rmtree
        except PermissionError:
            continue  # alive under another uid
        except OSError:
            continue
        shutil.rmtree(os.path.join(base, name), ignore_errors=True)
