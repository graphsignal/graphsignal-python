import logging
import sys
import os
import time
import asyncio
import unittest
import subprocess
from unittest.mock import Mock, patch

import graphsignal
import torch
from graphsignal.core.signal_uploader import SignalUploader
from graphsignal.core.ticker import Ticker
from graphsignal.utils import sha1
from test.test_utils import find_attribute, find_counter, find_tag

logger = logging.getLogger('graphsignal')


class SGLangRecorderTest(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None
        if len(logger.handlers) == 0:
            logger.addHandler(logging.StreamHandler(sys.stdout))
        graphsignal.configure(
            api_key='k1',
            debug_mode=True)
        graphsignal._ticker._auto_tick = False

    def tearDown(self):
        graphsignal.shutdown()
        try:
            subprocess.run(['pkill', '-f', 'sglang'], check=False, timeout=5)
        except Exception:
            pass

    @patch.object(SignalUploader, 'upload_span')
    @patch.object(Ticker, 'should_trace', return_value=True)
    def test_convert_otel_span(self, mocked_should_trace, mocked_upload_span):
        try:
            import sglang
        except ImportError:
            self.skipTest('sglang is not installed')
            return

        from graphsignal.recorders.sglang_recorder import SGLangRecorder

        recorder = SGLangRecorder()
        recorder._startup_options = {
            'tp_size': 2,
            'chunked_prefill_size': 4096,
            'attention_backend': 'flashinfer',
        }

        mock_otel_span = Mock()
        mock_otel_span.name = 'req_root'
        mock_otel_span.start_time = 1000000000
        mock_otel_span.end_time = 2000000000
        mock_otel_span.trace_id = 'traceabc123'
        mock_otel_span.span_id = 'spanabc123'
        mock_otel_span.parent_span_id = ''
        mock_otel_span.attributes = {
            'gen_ai.request.id': 'req_123',
            'gen_ai.response.model': 'Qwen/Qwen2.5-1.5B-Instruct',
            'gen_ai.request.temperature': 0.1,
            'gen_ai.request.top_p': 0.9,
            'gen_ai.request.top_k': 20,
            'gen_ai.request.max_tokens': 128,
            'gen_ai.request.n': 1,
            'gen_ai.usage.prompt_tokens': 32,
            'gen_ai.usage.cached_tokens': 8,
            'gen_ai.usage.completion_tokens': 16,
            'gen_ai.latency.time_in_queue': 0.020,
            'gen_ai.latency.time_to_first_token': 0.130,
            'gen_ai.latency.e2e': 0.500,
            'gen_ai.latency.time_in_model_prefill': 0.080,
            'gen_ai.latency.time_in_model_decode': 0.350,
            'gen_ai.latency.time_in_model_inference': 0.430,
        }

        recorder._convert_otel_span(mock_otel_span)

        mocked_upload_span.assert_called_once()
        span = mocked_upload_span.call_args[0][0]

        self.assertEqual(span.start_ts, 1000000000)
        self.assertEqual(span.end_ts, 2000000000)
        self.assertEqual(find_counter(span, 'span.duration'), 1000000000.0)
        self.assertEqual(span.name, 'sglang.req_root')
        self.assertEqual(span.trace_id, sha1('traceabc123', size=12))
        self.assertEqual(span.span_id, sha1('spanabc123', size=12))
        self.assertEqual(span.parent_span_id, '')
        self.assertEqual(find_tag(span, 'sampling.reason'), 'sglang.otel')

        self.assertEqual(find_tag(span, 'sglang.request.id'), 'req_123')
        self.assertEqual(find_tag(span, 'sglang.response.model'), 'Qwen/Qwen2.5-1.5B-Instruct')
        self.assertEqual(find_attribute(span, 'sglang.response.model'), 'Qwen/Qwen2.5-1.5B-Instruct')

        self.assertEqual(find_attribute(span, 'sglang.request.temperature'), '0.1')
        self.assertEqual(find_attribute(span, 'sglang.request.top_p'), '0.9')
        self.assertEqual(find_attribute(span, 'sglang.request.top_k'), '20')
        self.assertEqual(find_attribute(span, 'sglang.request.max_tokens'), '128')
        self.assertEqual(find_attribute(span, 'sglang.request.n'), '1')

        self.assertEqual(find_counter(span, 'sglang.usage.prompt_tokens'), 32.0)
        self.assertEqual(find_counter(span, 'sglang.usage.cached_tokens'), 8.0)
        self.assertEqual(find_counter(span, 'sglang.usage.completion_tokens'), 16.0)

        self.assertEqual(find_counter(span, 'sglang.latency.time_in_queue'), 20000000)
        self.assertEqual(find_counter(span, 'sglang.latency.time_to_first_token'), 130000000)
        self.assertEqual(find_counter(span, 'sglang.latency.e2e'), 500000000)
        self.assertEqual(find_counter(span, 'sglang.latency.time_in_model_prefill'), 80000000)
        self.assertEqual(find_counter(span, 'sglang.latency.time_in_model_decode'), 350000000)
        self.assertEqual(find_counter(span, 'sglang.latency.time_in_model_inference'), 430000000)

        self.assertIsNone(find_attribute(span, 'sglang.startup.tp_size'))
        self.assertIsNone(find_attribute(span, 'sglang.startup.chunked_prefill_size'))
        self.assertIsNone(find_attribute(span, 'sglang.startup.attention_backend'))

    @patch.object(SignalUploader, 'upload_span')
    @patch.object(Ticker, 'should_trace', return_value=True)
    def test_convert_otel_span_uses_latency_time_in_request_fallback(self, mocked_should_trace, mocked_upload_span):
        try:
            import sglang
        except ImportError:
            self.skipTest('sglang is not installed')
            return

        from graphsignal.recorders.sglang_recorder import SGLangRecorder

        recorder = SGLangRecorder()

        mock_otel_span = Mock()
        mock_otel_span.name = 'req_root'
        mock_otel_span.start_time = 1000000000
        mock_otel_span.end_time = 1500000000
        mock_otel_span.trace_id = 'trace_req_fallback'
        mock_otel_span.span_id = 'span_req_fallback'
        mock_otel_span.parent_span_id = ''
        mock_otel_span.attributes = {
            'gen_ai.latency.time_in_request': 0.25,
        }

        recorder._convert_otel_span(mock_otel_span)

        mocked_upload_span.assert_called_once()
        span = mocked_upload_span.call_args[0][0]
        self.assertEqual(find_counter(span, 'sglang.latency.e2e'), 250000000.0)

    @patch.object(SignalUploader, 'upload_span')
    @patch.object(Ticker, 'should_trace', return_value=True)
    def test_convert_otel_span_samples_root_and_children(self, mocked_should_trace, mocked_upload_span):
        try:
            import sglang
        except ImportError:
            self.skipTest('sglang is not installed')
            return

        from graphsignal.recorders.sglang_recorder import SGLangRecorder

        recorder = SGLangRecorder()

        root_span = Mock()
        root_span.name = 'req_root'
        root_span.start_time = 1000000000
        root_span.end_time = 2000000000
        root_span.trace_id = 'trace1'
        root_span.span_id = 'root1'
        root_span.parent_span_id = ''
        root_span.attributes = {}

        child_span = Mock()
        child_span.name = 'scheduler_step'
        child_span.start_time = 1200000000
        child_span.end_time = 1300000000
        child_span.trace_id = 'trace1'
        child_span.span_id = 'child1'
        child_span.parent_span_id = 'root1'
        child_span.attributes = {}

        recorder._convert_otel_span(root_span)
        recorder._convert_otel_span(child_span)

        self.assertEqual(mocked_upload_span.call_count, 2)
        mocked_should_trace.assert_called_once()

    @patch.object(SignalUploader, 'upload_span')
    @patch.object(Ticker, 'should_trace', return_value=False)
    def test_convert_otel_span_drops_children_when_root_not_sampled(self, mocked_should_trace, mocked_upload_span):
        try:
            import sglang
        except ImportError:
            self.skipTest('sglang is not installed')
            return

        from graphsignal.recorders.sglang_recorder import SGLangRecorder

        recorder = SGLangRecorder()

        root_span = Mock()
        root_span.name = 'req_root'
        root_span.start_time = 1000000000
        root_span.end_time = 2000000000
        root_span.trace_id = 'trace1'
        root_span.span_id = 'root1'
        root_span.parent_span_id = ''
        root_span.attributes = {}

        child_span = Mock()
        child_span.name = 'scheduler_step'
        child_span.start_time = 1200000000
        child_span.end_time = 1300000000
        child_span.trace_id = 'trace1'
        child_span.span_id = 'child1'
        child_span.parent_span_id = 'root1'
        child_span.attributes = {}

        recorder._convert_otel_span(root_span)
        recorder._convert_otel_span(child_span)

        mocked_upload_span.assert_not_called()
        mocked_should_trace.assert_called_once()

    @patch.object(SignalUploader, 'upload_span')
    @patch.object(Ticker, 'should_trace', return_value=True)
    def test_convert_otel_span_drops_when_ids_missing(self, mocked_should_trace, mocked_upload_span):
        try:
            import sglang
        except ImportError:
            self.skipTest('sglang is not installed')
            return

        from graphsignal.recorders.sglang_recorder import SGLangRecorder

        recorder = SGLangRecorder()
        span_without_ids = Mock()
        span_without_ids.name = 'req_root'
        span_without_ids.start_time = 1000000000
        span_without_ids.end_time = 2000000000
        span_without_ids.trace_id = ''
        span_without_ids.span_id = ''
        span_without_ids.parent_span_id = ''
        span_without_ids.attributes = {}

        recorder._convert_otel_span(span_without_ids)

        mocked_upload_span.assert_not_called()
        mocked_should_trace.assert_not_called()

    @unittest.skipIf(os.getenv('RUN_SGLANG_TESTS') != '1', 'skipped unless forced')
    @patch.object(SignalUploader, 'upload_span')
    def test_engine_generate(self, mocked_upload_span):
        try:
            import sglang
            from sglang.srt.server_args import ServerArgs
            from sglang.srt.entrypoints.engine import Engine
        except ImportError:
            self.skipTest('sglang is not installed')
            return

        if not torch.cuda.is_available():
            self.skipTest('No CUDA available')
            return

        # Ensure recorder is initialized via supported module import hook.
        _ = sglang.__version__
        sglang_recorder = None
        for recorder in graphsignal._ticker.recorders():
            if recorder.__class__.__name__ == 'SGLangRecorder':
                sglang_recorder = recorder
                break
        if sglang_recorder is None or not sglang_recorder._otel_endpoint:
            self.skipTest('SGLang recorder OTEL endpoint not available')
            return

        server_args = ServerArgs(
            model_path='distilgpt2',
            tp_size=1,
            log_level='error',
            enable_trace=True,
            enable_metrics=True,
            otlp_traces_endpoint=sglang_recorder._otel_endpoint,
        )

        engine = Engine(server_args=server_args)
        try:
            ret = engine.generate(
                prompt='What is 2 raised to the power of 10?',
                sampling_params={'temperature': 0.0, 'max_new_tokens': 16},
                stream=False,
            )
            self.assertIsNotNone(ret)

            # OTEL export is batched; allow callbacks to flush.
            for _ in range(12):
                if mocked_upload_span.called:
                    break
                time.sleep(0.25)

            self.assertTrue(mocked_upload_span.called)
            span = mocked_upload_span.call_args[0][0]
            self.assertTrue(span.name.startswith('sglang.'))
            self.assertEqual(find_tag(span, 'engine.name'), 'sglang')
            self.assertEqual(find_tag(span, 'engine.version'), sglang.__version__)
            self.assertEqual(find_tag(span, 'sampling.reason'), 'sglang.otel')
        finally:
            engine.shutdown()

    @unittest.skipIf(os.getenv('RUN_SGLANG_TESTS') != '1', 'skipped unless forced')
    @patch.object(SignalUploader, 'upload_span')
    def test_engine_async_generate(self, mocked_upload_span):
        try:
            import sglang
            from sglang.srt.server_args import ServerArgs
            from sglang.srt.entrypoints.engine import Engine
        except ImportError:
            self.skipTest('sglang is not installed')
            return

        if not torch.cuda.is_available():
            self.skipTest('No CUDA available')
            return

        _ = sglang.__version__
        sglang_recorder = None
        for recorder in graphsignal._ticker.recorders():
            if recorder.__class__.__name__ == 'SGLangRecorder':
                sglang_recorder = recorder
                break
        if sglang_recorder is None or not sglang_recorder._otel_endpoint:
            self.skipTest('SGLang recorder OTEL endpoint not available')
            return

        server_args = ServerArgs(
            model_path='distilgpt2',
            tp_size=1,
            log_level='error',
            enable_trace=True,
            enable_metrics=True,
            otlp_traces_endpoint=sglang_recorder._otel_endpoint,
        )

        engine = Engine(server_args=server_args)
        try:
            async def _run_async():
                stream_iter = await engine.async_generate(
                    prompt='Give one short sentence about machine learning.',
                    sampling_params={'temperature': 0.0, 'max_new_tokens': 16},
                    stream=True,
                )
                chunks = []
                async for chunk in stream_iter:
                    chunks.append(chunk)
                return chunks

            chunks = asyncio.run(_run_async())
            self.assertTrue(len(chunks) > 0)

            for _ in range(12):
                if mocked_upload_span.called:
                    break
                time.sleep(0.25)

            self.assertTrue(mocked_upload_span.called)
            span = mocked_upload_span.call_args[0][0]
            self.assertTrue(span.name.startswith('sglang.'))
            self.assertEqual(find_tag(span, 'engine.name'), 'sglang')
            self.assertEqual(find_tag(span, 'engine.version'), sglang.__version__)
            self.assertEqual(find_tag(span, 'sampling.reason'), 'sglang.otel')
        finally:
            engine.shutdown()


if __name__ == '__main__':
    unittest.main()
