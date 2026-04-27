import unittest
import logging
import sys
import os
import time
from unittest.mock import patch, Mock
import subprocess
import torch

import graphsignal
from graphsignal.proto import signals_pb2
from graphsignal.core.signal_uploader import SignalUploader
from graphsignal.core.ticker import Ticker
from graphsignal.profilers.event_profiler import EventProfiler
from graphsignal.utils import sha1
from test.test_utils import find_tag, find_attribute, find_counter
from test.core.test_signal_uploader import HttpTestServer

logger = logging.getLogger('graphsignal')


class VLLMRecorderTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.maxDiff = None
        if len(logger.handlers) == 0:
            logger.addHandler(logging.StreamHandler(sys.stdout))
        graphsignal.configure(
            api_key='k1',
            debug_mode=True)
        graphsignal._ticker._auto_tick = False

    async def asyncTearDown(self):
        graphsignal.shutdown()

        try:
            subprocess.run(['pkill', '-f', 'VLLM'], check=False, timeout=5)
        except Exception:
            pass

    @patch.object(SignalUploader, 'upload_span')
    @patch.object(Ticker, 'should_trace', return_value=True)
    def test_convert_otel_span(self, mocked_should_trace, mocked_upload_span):
        try:
            import vllm
        except ImportError:
            self.skipTest("vllm is not installed")
            return
            
        from graphsignal.recorders.vllm_recorder import VLLMRecorder
        
        recorder = VLLMRecorder()
        recorder._startup_options = {
            'tensor_parallel_size': 2,
            'dtype': 'bfloat16',
            'enforce_eager': False,
        }
        
        mock_otel_span = Mock()
        mock_otel_span.name = "llm_request"
        mock_otel_span.start_time = 1000000000
        mock_otel_span.end_time = 2000000000
        mock_otel_span.trace_id = 'trace_vllm_123'
        mock_otel_span.span_id = 'span_vllm_123'
        mock_otel_span.parent_span_id = ''
        mock_otel_span.attributes = {
            'gen_ai.request.id': 'test_request_123',
            'gen_ai.response.model': 'Qwen/Qwen1.5-7B-Chat',
            'gen_ai.request.temperature': 0.7,
            'gen_ai.request.top_p': 0.95,
            'gen_ai.request.max_tokens': 256,
            'gen_ai.request.n': 1,
            'gen_ai.usage.num_sequences': 1,
            'gen_ai.usage.prompt_tokens': 16,
            'gen_ai.usage.completion_tokens': 1,
            'gen_ai.latency.time_in_queue': 0.012926340103149414,
            'gen_ai.latency.time_to_first_token': 0.22484421730041504,
            'gen_ai.latency.e2e': 0.22525858879089355,
            'gen_ai.latency.time_in_scheduler': 0.0013919062912464142
        }
        
        recorder._convert_otel_span(mock_otel_span)
        
        mocked_upload_span.assert_called_once()
        span = mocked_upload_span.call_args[0][0]

        self.assertEqual(span.start_ts, 1000000000)
        self.assertEqual(span.end_ts, 2000000000)
        self.assertEqual(find_counter(span, 'span.duration'), 1000000000.0)
        self.assertEqual(span.name, 'vllm.llm_request')
        self.assertEqual(span.trace_id, sha1('trace_vllm_123', size=12))
        self.assertEqual(span.span_id, sha1('span_vllm_123', size=12))
        self.assertEqual(span.parent_span_id, '')

        self.assertEqual(find_tag(span, 'sampling.reason'), 'vllm.otel')
        
        self.assertEqual(find_tag(span, 'vllm.request.id'), 'test_request_123')
        
        self.assertEqual(find_tag(span, 'vllm.response.model'), 'Qwen/Qwen1.5-7B-Chat')
        self.assertEqual(find_attribute(span, 'vllm.response.model'), 'Qwen/Qwen1.5-7B-Chat')
        
        self.assertEqual(find_attribute(span, 'vllm.request.temperature'), '0.7')
        self.assertEqual(find_attribute(span, 'vllm.request.top_p'), '0.95')
        self.assertEqual(find_attribute(span, 'vllm.request.max_tokens'), '256')
        self.assertEqual(find_attribute(span, 'vllm.request.n'), '1')
        
        self.assertEqual(find_counter(span, 'vllm.usage.num_sequences'), 1.0)
        self.assertEqual(find_counter(span, 'vllm.usage.prompt_tokens'), 16.0)
        self.assertEqual(find_counter(span, 'vllm.usage.completion_tokens'), 1.0)
        
        self.assertEqual(find_counter(span, 'vllm.latency.time_in_queue'), 12926340)
        self.assertEqual(find_counter(span, 'vllm.latency.time_to_first_token'), 224844217)
        self.assertEqual(find_counter(span, 'vllm.latency.e2e'), 225258588)
        self.assertEqual(find_counter(span, 'vllm.latency.time_in_scheduler'), 1391906)

        self.assertIsNone(find_attribute(span, 'vllm.startup.tensor_parallel_size'))
        self.assertIsNone(find_attribute(span, 'vllm.startup.dtype'))
        self.assertIsNone(find_attribute(span, 'vllm.startup.enforce_eager'))

    @patch.object(SignalUploader, 'upload_span')
    @patch.object(Ticker, 'should_trace', return_value=True)
    def test_convert_otel_span_samples_root_and_children(self, mocked_should_trace, mocked_upload_span):
        try:
            import vllm
        except ImportError:
            self.skipTest("vllm is not installed")
            return

        from graphsignal.recorders.vllm_recorder import VLLMRecorder

        recorder = VLLMRecorder()

        root_span = Mock()
        root_span.name = "llm_request"
        root_span.start_time = 1000000000
        root_span.end_time = 2000000000
        root_span.trace_id = 'tracev1'
        root_span.span_id = 'rootv1'
        root_span.parent_span_id = ''
        root_span.attributes = {}

        child_span = Mock()
        child_span.name = "model_step"
        child_span.start_time = 1200000000
        child_span.end_time = 1300000000
        child_span.trace_id = 'tracev1'
        child_span.span_id = 'childv1'
        child_span.parent_span_id = 'rootv1'
        child_span.attributes = {}

        recorder._convert_otel_span(root_span)
        recorder._convert_otel_span(child_span)

        self.assertEqual(mocked_upload_span.call_count, 2)
        mocked_should_trace.assert_called_once()

    @patch.object(SignalUploader, 'upload_span')
    @patch.object(Ticker, 'should_trace', return_value=False)
    def test_convert_otel_span_drops_children_when_root_not_sampled(self, mocked_should_trace, mocked_upload_span):
        try:
            import vllm
        except ImportError:
            self.skipTest("vllm is not installed")
            return

        from graphsignal.recorders.vllm_recorder import VLLMRecorder

        recorder = VLLMRecorder()

        root_span = Mock()
        root_span.name = "llm_request"
        root_span.start_time = 1000000000
        root_span.end_time = 2000000000
        root_span.trace_id = 'tracev1'
        root_span.span_id = 'rootv1'
        root_span.parent_span_id = ''
        root_span.attributes = {}

        child_span = Mock()
        child_span.name = "model_step"
        child_span.start_time = 1200000000
        child_span.end_time = 1300000000
        child_span.trace_id = 'tracev1'
        child_span.span_id = 'childv1'
        child_span.parent_span_id = 'rootv1'
        child_span.attributes = {}

        recorder._convert_otel_span(root_span)
        recorder._convert_otel_span(child_span)

        mocked_upload_span.assert_not_called()
        mocked_should_trace.assert_called_once()

    @patch.object(SignalUploader, 'upload_span')
    @patch.object(Ticker, 'should_trace', return_value=True)
    def test_convert_otel_span_drops_when_ids_missing(self, mocked_should_trace, mocked_upload_span):
        try:
            import vllm
        except ImportError:
            self.skipTest("vllm is not installed")
            return

        from graphsignal.recorders.vllm_recorder import VLLMRecorder

        recorder = VLLMRecorder()
        span_without_ids = Mock()
        span_without_ids.name = "llm_request"
        span_without_ids.start_time = 1000000000
        span_without_ids.end_time = 2000000000
        span_without_ids.trace_id = ''
        span_without_ids.span_id = ''
        span_without_ids.parent_span_id = ''
        span_without_ids.attributes = {}

        recorder._convert_otel_span(span_without_ids)

        mocked_upload_span.assert_not_called()
        mocked_should_trace.assert_not_called()


    @patch.object(EventProfiler, 'record_event')
    @patch.object(SignalUploader, 'upload_span')
    @patch.object(Ticker, 'should_trace', return_value=True)
    def test_convert_otel_span_calls_record_event(self, mocked_should_trace, mocked_upload_span, mocked_record_event):
        try:
            import vllm
        except ImportError:
            self.skipTest("vllm is not installed")
            return

        from graphsignal.recorders.vllm_recorder import VLLMRecorder

        recorder = VLLMRecorder()

        mock_otel_span = Mock()
        mock_otel_span.name = 'llm_request'
        mock_otel_span.start_time = 1000000000
        mock_otel_span.end_time = 2000000000
        mock_otel_span.trace_id = 'trace_vllm_re1'
        mock_otel_span.span_id = 'span_vllm_re1'
        mock_otel_span.parent_span_id = ''
        mock_otel_span.attributes = {}

        recorder._convert_otel_span(mock_otel_span)

        mocked_record_event.assert_called_once()
        kw = mocked_record_event.call_args.kwargs
        self.assertEqual(kw['op_name'], 'vllm.llm_request')
        self.assertEqual(kw['category'], 'vllm.otel')
        self.assertEqual(kw['start_ns'], 1000000000)
        self.assertEqual(kw['end_ns'], 2000000000)

    @patch.object(EventProfiler, 'record_event')
    @patch.object(SignalUploader, 'upload_span')
    @patch.object(Ticker, 'should_trace', return_value=False)
    def test_convert_otel_span_no_record_event_when_not_sampled(self, mocked_should_trace, mocked_upload_span, mocked_record_event):
        try:
            import vllm
        except ImportError:
            self.skipTest("vllm is not installed")
            return

        from graphsignal.recorders.vllm_recorder import VLLMRecorder

        recorder = VLLMRecorder()

        mock_otel_span = Mock()
        mock_otel_span.name = 'llm_request'
        mock_otel_span.start_time = 1000000000
        mock_otel_span.end_time = 2000000000
        mock_otel_span.trace_id = 'trace_vllm_re2'
        mock_otel_span.span_id = 'span_vllm_re2'
        mock_otel_span.parent_span_id = ''
        mock_otel_span.attributes = {}

        recorder._convert_otel_span(mock_otel_span)

        mocked_record_event.assert_not_called()

    @unittest.skipIf(os.getenv("RUN_VLLM_TESTS") != "1", "skipped unless forced")
    @patch.object(SignalUploader, 'upload_span')
    def test_llm_generate(self, mocked_upload_span):
        try:
            import vllm
        except ImportError:
            self.skipTest("vllm is not installed")
            return

        if not torch.cuda.is_available():
            self.skipTest("No CUDA available")
            return

        _ = vllm.__version__
        vllm_recorder = None
        for recorder in graphsignal._ticker.recorders():
            if recorder.__class__.__name__ == 'VLLMRecorder':
                vllm_recorder = recorder
                break
        if vllm_recorder is None or not vllm_recorder._otel_endpoint:
            self.skipTest('vLLM recorder OTEL endpoint not available')
            return

        from vllm import LLM, SamplingParams

        llm = LLM(model="distilgpt2")

        sampling_params = SamplingParams(
            temperature=0.7, 
            top_p=0.95, 
            max_tokens=256
        )

        outputs = llm.generate(["What is 2 raised to 10 power?"], sampling_params)
        outputs = llm.generate(["What is 2 raised to 10 power?"], sampling_params)

        for _ in range(12):
            if mocked_upload_span.called:
                break
            time.sleep(0.25)

        self.assertTrue(mocked_upload_span.called)
        span = mocked_upload_span.call_args[0][0]

        self.assertTrue(span.name.startswith('vllm.'))
        self.assertEqual(find_tag(span, 'engine.name'), 'vllm')
        self.assertEqual(find_tag(span, 'engine.version'), vllm.__version__)
        self.assertEqual(find_tag(span, 'sampling.reason'), 'vllm.otel')


    @unittest.skipIf(os.getenv("RUN_VLLM_TESTS") != "1", "skipped unless forced")
    @patch.object(SignalUploader, 'upload_span')
    async def test_async_llm_generate(self, mocked_upload_span):
        try:
            import vllm
        except ImportError:
            self.skipTest("vllm is not installed")
            return

        if not torch.cuda.is_available():
            self.skipTest("No CUDA available")
            return

        _ = vllm.__version__
        vllm_recorder = None
        for recorder in graphsignal._ticker.recorders():
            if recorder.__class__.__name__ == 'VLLMRecorder':
                vllm_recorder = recorder
                break
        if vllm_recorder is None or not vllm_recorder._otel_endpoint:
            self.skipTest('vLLM recorder OTEL endpoint not available')
            return

        from vllm.v1.engine.async_llm import AsyncLLM
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm import SamplingParams

        engine_args = AsyncEngineArgs(model="distilgpt2")
        async_llm = AsyncLLM.from_engine_args(engine_args)

        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.95,
            max_tokens=256
        )

        prompt = "What is 2 raised to 10 power?"
        request_id = "test_async_request_1"

        outputs = []
        async for output in async_llm.generate(prompt, sampling_params, request_id):
            outputs.append(output)

        for _ in range(12):
            if mocked_upload_span.called:
                break
            time.sleep(0.25)

        self.assertTrue(mocked_upload_span.called)
        span = mocked_upload_span.call_args[0][0]

        self.assertTrue(span.name.startswith('vllm.'))
        self.assertEqual(find_tag(span, 'engine.name'), 'vllm')
        self.assertEqual(find_tag(span, 'engine.version'), vllm.__version__)
        self.assertEqual(find_tag(span, 'sampling.reason'), 'vllm.otel')
