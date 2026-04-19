import logging
import threading
import time
from typing import Any, Dict, Optional, Tuple

import graphsignal

logger = logging.getLogger('graphsignal')

MAX_EVENT_PROFILER_FIELDS = 250

DescriptorKey = Tuple[Tuple[str, str], ...]


class EventBucket:
    __slots__ = (
        'num_running',
        'enter_offset_ns',
        'exit_offset_ns',
        'extra',
    )

    def __init__(self):
        self.num_running = 0
        self.enter_offset_ns = 0
        self.exit_offset_ns = 0
        self.extra: Dict[str, int] = {}


def _descriptor_field_key(descriptor: Dict[str, Any]) -> DescriptorKey:
    return tuple(sorted((str(k), str(descriptor[k])) for k in descriptor))


class EventProfiler:
    """
    Aggregates custom timed events into resolution-aligned buckets and exports
    profile counters per descriptor. ``cumtime`` is computed from timing and
    distributed across all buckets the interval spans (start_ns to end_ns).
    All other stats are assigned to the terminal bucket (where end_ns falls).
    The field map for a descriptor is created once on the first ``record_event``
    call and never updated.
    """

    def __init__(self, profile_name: str):
        self._profile_name = profile_name
        self._resolution_ns = 10_000_000
        self._disabled = True
        self._fields: Dict[DescriptorKey, Dict[str, int]] = {}
        self._field_count = 0
        self._buckets: Dict[int, Dict[DescriptorKey, EventBucket]] = {}
        self._bucket_lock = threading.Lock()
        self._current_bucket_ts: Optional[int] = None
        self._rollover_stop_event: Optional[threading.Event] = None
        self._rollover_timer_thread: Optional[threading.Thread] = None

    def set_resolution_ns(self, resolution_ns: int) -> None:
        if resolution_ns < 10_000_000:
            resolution_ns = 10_000_000
        self._resolution_ns = resolution_ns

    def get_resolution_ns(self) -> int:
        return self._resolution_ns

    def setup(self) -> None:
        self._start_rollover_timer()
        self._current_bucket_ts = time.time_ns()
        self._disabled = False

    def shutdown(self) -> None:
        if self._disabled:
            return
        self._stop_rollover_timer()
        with self._bucket_lock:
            self._buckets.clear()
        self._fields.clear()
        self._field_count = 0
        self._disabled = True

    def _build_field_descriptor(
            self,
            descriptor: Dict[str, Any],
            statistic: str,
            unit: Optional[str] = None) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for key, value in descriptor.items():
            out[str(key)] = str(value)
        out['statistic'] = statistic
        if unit is not None:
            out['unit'] = unit
        return out

    def _ensure_descriptor_field_map(
            self,
            descriptor: Dict[str, Any],
            stats: Dict[str, Any]) -> Optional[Dict[str, int]]:
        key = _descriptor_field_key(descriptor)
        existing = self._fields.get(key)
        if existing is not None:
            return existing
        if self._field_count + 1 + len(stats) > MAX_EVENT_PROFILER_FIELDS:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    'Event profiler field limit reached (%s), skipping descriptor',
                    MAX_EVENT_PROFILER_FIELDS)
            return None
        field_map: Dict[str, int] = {}
        cum_id = graphsignal._ticker.add_counter_profile_field(
            descriptor=self._build_field_descriptor(descriptor, 'cumtime', unit='ns'))
        field_map['cumtime'] = cum_id
        self._field_count += 1
        for stat_name in stats:
            field_id = graphsignal._ticker.add_counter_profile_field(
                descriptor=self._build_field_descriptor(descriptor, stat_name))
            field_map[stat_name] = field_id
            self._field_count += 1
        self._fields[key] = field_map
        return field_map

    def record_event(
            self,
            descriptor: Dict[str, Any],
            stats: Dict[str, Any],
            start_ns: int,
            end_ns: int) -> None:
        if self._disabled:
            return
        if descriptor.get('op_name') is None or descriptor.get('category') is None:
            logger.error('record_event: descriptor must include op_name and category')
            return
        if not stats:
            logger.error('record_event: stats must include at least one stat')
            return

        if self._ensure_descriptor_field_map(descriptor, stats) is None:
            return

        values: Dict[str, int] = {}
        for stat_name, raw in stats.items():
            try:
                values[stat_name] = int(raw)
            except (TypeError, ValueError):
                logger.error('record_event: stat %s must be an integer', stat_name)
                return

        key = _descriptor_field_key(descriptor)
        with self._bucket_lock:
            self._add_event_interval(
                event_key=key,
                start_ts=start_ns,
                end_ts=end_ns,
                values=values,
            )

    def _align_down(self, ts_ns: int) -> int:
        res = self._resolution_ns
        return (ts_ns // res) * res

    def _add_event_interval(
            self,
            event_key: DescriptorKey,
            start_ts: int,
            end_ts: int,
            values: Dict[str, int]) -> None:
        if end_ts <= start_ts or self._resolution_ns == 0:
            return

        res = self._resolution_ns
        start_bucket = self._align_down(start_ts)
        end_bucket = self._align_down(end_ts - 1)

        bucket_ts = start_bucket
        while bucket_ts <= end_bucket:
            bucket_end = bucket_ts + res
            if bucket_ts not in self._buckets:
                self._buckets[bucket_ts] = {}
            events = self._buckets[bucket_ts]
            if event_key not in events:
                events[event_key] = EventBucket()
            eb = events[event_key]

            if bucket_ts == start_bucket:
                eb.enter_offset_ns += start_ts - bucket_ts

            if end_ts <= bucket_end:
                eb.exit_offset_ns += end_ts - bucket_ts
                for name, value in values.items():
                    eb.extra[name] = eb.extra.get(name, 0) + value
                break
            eb.num_running += 1
            bucket_ts = bucket_end

    def _start_rollover_timer(self) -> None:
        self._rollover_stop_event = threading.Event()

        def round_to_rollup(ts_ns: int) -> int:
            return ts_ns // self._resolution_ns * self._resolution_ns

        def _rollover_loop():
            while not self._rollover_stop_event.wait(self._resolution_ns / 1e9 / 10):
                try:
                    current_ts = self._current_bucket_ts
                    if current_ts is None:
                        continue
                    now_ns = time.time_ns()
                    if round_to_rollup(now_ns) > round_to_rollup(current_ts):
                        self._rollover_buckets(now_ns)
                except Exception as exc:
                    logger.error('Error in event profiler rollover timer: %s', exc, exc_info=True)

        self._rollover_timer_thread = threading.Thread(target=_rollover_loop, daemon=True)
        self._rollover_timer_thread.start()

    def _stop_rollover_timer(self) -> None:
        if self._rollover_timer_thread:
            assert self._rollover_stop_event is not None
            self._rollover_stop_event.set()
            self._rollover_timer_thread.join()
            self._rollover_stop_event = None
            self._rollover_timer_thread = None

    def _rollover_buckets(self, now_ns: int) -> None:
        if self._disabled:
            return

        def round_to_rollup(ts_ns: int) -> int:
            return ts_ns // self._resolution_ns * self._resolution_ns

        aligned_now = round_to_rollup(now_ns)
        res = self._resolution_ns

        profiles_by_ts: Dict[int, Dict[int, int]] = {}
        with self._bucket_lock:
            to_emit = sorted(bt for bt in self._buckets if bt < aligned_now)
            for bucket_ts in to_emit:
                events = self._buckets.pop(bucket_ts, None)
                if not events:
                    continue
                profile: Dict[int, int] = {}
                for event_key, bucket in events.items():
                    field_map = self._fields.get(event_key)
                    if not field_map:
                        continue
                    if bucket.num_running > 0 or bucket.exit_offset_ns > 0:
                        cumtime = (
                            res * bucket.num_running
                            - bucket.enter_offset_ns
                            + bucket.exit_offset_ns)
                        cumtime = max(0, int(cumtime))
                        fid = field_map.get('cumtime')
                        if fid and cumtime > 0:
                            profile[fid] = cumtime
                    for stat_name, total in bucket.extra.items():
                        fid = field_map.get(stat_name)
                        if fid and total > 0:
                            profile[fid] = total

                if profile:
                    profiles_by_ts[bucket_ts] = profile

        process_tags = graphsignal._ticker.process_tags()
        for bucket_ts in sorted(profiles_by_ts):
            graphsignal._ticker.update_profile(
                name=self._profile_name,
                profile=profiles_by_ts[bucket_ts],
                measurement_ts=bucket_ts + res,
                tags=process_tags,
            )

        self._current_bucket_ts = now_ns
