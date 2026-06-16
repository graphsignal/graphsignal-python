from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence

_INPUT_TOKEN_ATTRS: Sequence[str] = (
    'gen_ai.usage.input_tokens',
    'gen_ai.usage.prompt_tokens',
    'vllm.usage.prompt_tokens',
    'sglang.usage.prompt_tokens',
    'trtllm.usage.prompt_tokens',
)

_OUTPUT_TOKEN_ATTRS: Sequence[str] = (
    'gen_ai.usage.output_tokens',
    'gen_ai.usage.completion_tokens',
    'vllm.usage.completion_tokens',
    'sglang.usage.completion_tokens',
    'trtllm.usage.completion_tokens',
)

_CACHED_TOKEN_ATTRS: Sequence[str] = (
    'gen_ai.usage.cached_tokens',
    'vllm.usage.cached_tokens',
    'sglang.usage.cached_tokens',
    'trtllm.usage.cached_tokens',
)

_TTFT_ATTRS: Sequence[str] = (
    'gen_ai.latency.time_to_first_token',
    'vllm.latency.time_to_first_token',
    'sglang.latency.time_to_first_token',
    'trtllm.latency.time_to_first_token',
)

_PREFILL_ATTRS: Sequence[str] = (
    'gen_ai.latency.time_in_model_prefill',
    'vllm.latency.time_in_model_prefill',
    'sglang.latency.time_in_model_prefill',
    'trtllm.latency.time_in_model_prefill',
)


@dataclass(frozen=True)
class SpanTokenStats:
    input_tokens: int
    output_tokens: int
    cached_tokens: int
    phase_latency_ns: int


def _first_attr(attributes: Mapping[str, Any], keys: Sequence[str]) -> Optional[Any]:
    for key in keys:
        if key in attributes:
            return attributes[key]
    return None


def _parse_token_count(value: Any) -> int:
    if value is None:
        return 0
    try:
        count = int(float(value))
    except (TypeError, ValueError):
        return 0
    return max(count, 0)


def _parse_latency_ns(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    if seconds <= 0:
        return None
    return int(seconds * 1_000_000_000)


def extract_span_token_stats(attributes: Dict[str, Any]) -> Optional[SpanTokenStats]:
    if not attributes:
        return None

    phase_latency_ns = _parse_latency_ns(_first_attr(attributes, _TTFT_ATTRS))
    if phase_latency_ns is None:
        phase_latency_ns = _parse_latency_ns(_first_attr(attributes, _PREFILL_ATTRS))
    if phase_latency_ns is None:
        return None

    input_tokens = _parse_token_count(_first_attr(attributes, _INPUT_TOKEN_ATTRS))
    output_tokens = _parse_token_count(_first_attr(attributes, _OUTPUT_TOKEN_ATTRS))
    cached_tokens = _parse_token_count(_first_attr(attributes, _CACHED_TOKEN_ATTRS))

    if input_tokens == 0 and output_tokens == 0 and cached_tokens == 0:
        return None

    return SpanTokenStats(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_tokens=cached_tokens,
        phase_latency_ns=phase_latency_ns,
    )
