"""Derive stable SpanProfiler op_name from OTLP span name + attributes.

Full span names are preserved in SpanStore for trace drill-down; only the
profile.events op_name is normalized so per-request / per-process suffixes
collapse into one field per logical operation.
"""

import re
from typing import Any, Mapping, Optional

_SGLANG_SERVICE_PREFIX = 'sglang'
_SGLANG_NAME_PREFIXES = ('sglang.', 'sglang-diffusion.')

_THREAD_HOST_PID_SUFFIX = re.compile(
    r' \(host:[0-9a-f]+ \| pid:\d+\)$')
_REQUEST_ID_SUFFIX = re.compile(
    r'(\.?)\s*Req [0-9a-f]{8}$', re.IGNORECASE)


def stable_otel_op_name(
        span_name: str,
        attributes: Optional[Mapping[str, Any]] = None) -> str:
    """Return a stable op_name for profile.events aggregation."""
    if not span_name:
        return span_name

    attrs = attributes or {}
    if not _should_apply_sglang_normalize(span_name, attrs):
        return span_name

    thread_label = attrs.get('thread_label')
    if thread_label is not None:
        return _thread_op_name(span_name, attrs)

    return _sglang_pattern_normalize(span_name)


def _should_apply_sglang_normalize(
        span_name: str,
        attrs: Mapping[str, Any]) -> bool:
    svc = str(attrs.get('service.name', ''))
    if svc.startswith(_SGLANG_SERVICE_PREFIX):
        return True
    if str(attrs.get('module', '')).startswith('sglang::'):
        return True
    return span_name.startswith(_SGLANG_NAME_PREFIXES)


def _engine_prefix(span_name: str, attrs: Mapping[str, Any]) -> str:
    engine = attrs.get('service.name')
    if engine:
        return f'{engine}.'
    dot = span_name.find('.')
    if 0 < dot < len(span_name) - 1:
        return span_name[:dot + 1]
    return ''


def _thread_op_name(span_name: str, attrs: Mapping[str, Any]) -> str:
    prefix = _engine_prefix(span_name, attrs)
    label = str(attrs['thread_label'])
    parts = [label]
    for key, tag in (('tp_rank', 'TP'), ('pp_rank', 'PP'), ('dp_rank', 'DP')):
        rank = attrs.get(key)
        if rank is not None:
            parts.append(f'[{tag} {rank}]')
    return prefix + ' '.join(parts)


def _sglang_pattern_normalize(span_name: str) -> str:
    name = _THREAD_HOST_PID_SUFFIX.sub('', span_name)
    m = _REQUEST_ID_SUFFIX.search(name)
    if not m:
        return name
    prefix = name[:m.start()]
    if prefix.endswith('.'):
        return prefix + 'Req'
    return f'{prefix}.Req' if prefix else 'Req'
