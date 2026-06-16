import unittest

from graphsignal.otel.span_token_stats import extract_span_token_stats


class SpanTokenStatsTest(unittest.TestCase):
    def test_extract_gen_ai_attributes(self):
        stats = extract_span_token_stats({
            'gen_ai.usage.prompt_tokens': 100,
            'gen_ai.usage.completion_tokens': 60,
            'gen_ai.usage.cached_tokens': 30,
            'gen_ai.latency.time_to_first_token': 0.04,
        })
        self.assertIsNotNone(stats)
        assert stats is not None
        self.assertEqual(stats.input_tokens, 100)
        self.assertEqual(stats.output_tokens, 60)
        self.assertEqual(stats.cached_tokens, 30)
        self.assertEqual(stats.phase_latency_ns, 40_000_000)

    def test_extract_input_output_semconv_aliases(self):
        stats = extract_span_token_stats({
            'gen_ai.usage.input_tokens': 50,
            'gen_ai.usage.output_tokens': 20,
            'gen_ai.latency.time_in_model_prefill': 0.1,
        })
        self.assertIsNotNone(stats)
        assert stats is not None
        self.assertEqual(stats.input_tokens, 50)
        self.assertEqual(stats.output_tokens, 20)
        self.assertEqual(stats.cached_tokens, 0)
        self.assertEqual(stats.phase_latency_ns, 100_000_000)

    def test_extract_vllm_legacy_aliases(self):
        stats = extract_span_token_stats({
            'vllm.usage.prompt_tokens': 10,
            'vllm.usage.completion_tokens': 5,
            'vllm.latency.time_to_first_token': 0.02,
        })
        self.assertIsNotNone(stats)
        assert stats is not None
        self.assertEqual(stats.input_tokens, 10)
        self.assertEqual(stats.output_tokens, 5)

    def test_extract_sglang_legacy_aliases(self):
        stats = extract_span_token_stats({
            'sglang.usage.prompt_tokens': 12,
            'sglang.usage.completion_tokens': 8,
            'sglang.usage.cached_tokens': 4,
            'sglang.latency.time_in_model_prefill': 0.03,
        })
        self.assertIsNotNone(stats)
        assert stats is not None
        self.assertEqual(stats.input_tokens, 12)
        self.assertEqual(stats.output_tokens, 8)
        self.assertEqual(stats.cached_tokens, 4)

    def test_extract_trtllm_fixture(self):
        stats = extract_span_token_stats({
            'service.name': 'trtllm-server',
            'gen_ai.usage.prompt_tokens': 200,
            'gen_ai.usage.completion_tokens': 80,
            'gen_ai.latency.time_to_first_token': 0.05,
        })
        self.assertIsNotNone(stats)
        assert stats is not None
        self.assertEqual(stats.input_tokens, 200)
        self.assertEqual(stats.output_tokens, 80)
        self.assertEqual(stats.cached_tokens, 0)
        self.assertEqual(stats.phase_latency_ns, 50_000_000)

    def test_returns_none_without_phase_latency(self):
        self.assertIsNone(extract_span_token_stats({
            'gen_ai.usage.prompt_tokens': 100,
            'gen_ai.usage.completion_tokens': 60,
        }))
        self.assertIsNone(extract_span_token_stats({
            'gen_ai.usage.prompt_tokens': 100,
            'gen_ai.usage.completion_tokens': 60,
            'gen_ai.latency.time_to_first_token': 0,
        }))

    def test_returns_none_when_all_token_counts_zero(self):
        self.assertIsNone(extract_span_token_stats({
            'gen_ai.latency.time_to_first_token': 0.04,
        }))

    def test_ttft_preferred_over_prefill(self):
        stats = extract_span_token_stats({
            'gen_ai.usage.prompt_tokens': 10,
            'gen_ai.latency.time_to_first_token': 0.04,
            'gen_ai.latency.time_in_model_prefill': 0.1,
        })
        self.assertIsNotNone(stats)
        assert stats is not None
        self.assertEqual(stats.phase_latency_ns, 40_000_000)

    def test_empty_attributes(self):
        self.assertIsNone(extract_span_token_stats({}))
        self.assertIsNone(extract_span_token_stats(None))  # type: ignore[arg-type]


if __name__ == '__main__':
    unittest.main()
