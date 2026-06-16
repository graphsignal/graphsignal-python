import unittest
import sys
import time
import logging
from unittest.mock import MagicMock

import graphsignal
import graphsignal.sdk
from graphsignal.otel.otel_collector import OTELCollector, OTELCollectorServicer
from test.test_utils import find_tag, find_attribute, find_counter

logger = logging.getLogger('graphsignal')


def _make_trace_id_bytes(seed: int) -> bytes:
    return seed.to_bytes(16, 'big')


def _make_span_id_bytes(seed: int) -> bytes:
    return seed.to_bytes(8, 'big')


def _make_otlp_attr(key, *, int_value=None, double_value=None):
    attr = MagicMock()
    attr.key = key
    value = MagicMock()
    if int_value is not None:
        value.WhichOneof.return_value = 'int_value'
        value.int_value = int_value
    elif double_value is not None:
        value.WhichOneof.return_value = 'double_value'
        value.double_value = double_value
    else:
        value.WhichOneof.return_value = None
    attr.value = value
    return attr


class OTELCollectorTest(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None
        if len(logger.handlers) == 0:
            logger.addHandler(logging.StreamHandler(sys.stdout))
        graphsignal.sdk.configure(
            api_key='k1',
            debug_mode=True)
        graphsignal.sdk.sdk()._auto_tick = False

    def tearDown(self):
        graphsignal.sdk.shutdown()

    def test_servicer_records_span_in_span_store(self):
        try:
            import grpc
            from opentelemetry.proto.collector.trace.v1.trace_service_pb2_grpc import TraceServiceStub
            from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
            from opentelemetry.proto.trace.v1.trace_pb2 import ResourceSpans, ScopeSpans, Span as OtelSpan
            from opentelemetry.proto.resource.v1.resource_pb2 import Resource
            from opentelemetry.proto.common.v1.common_pb2 import InstrumentationScope
        except ImportError:
            self.skipTest("gRPC / OTLP protobufs not available")

        # Force the span store to always sample so the test is deterministic.
        graphsignal.sdk.sdk().config_loader()._options['traces_per_sec'] = '1000'

        collector = OTELCollector(port=OTELCollector.find_port())
        collector.setup()
        try:
            endpoint = collector.get_endpoint()
            if not endpoint:
                self.skipTest("OTEL collector did not start")

            port = endpoint.split(':')[-1]
            channel = grpc.insecure_channel(f'localhost:{port}')
            stub = TraceServiceStub(channel)

            span = OtelSpan()
            span.name = "test_span"
            span.trace_id = _make_trace_id_bytes(0xA1)
            span.span_id = _make_span_id_bytes(0xB1)
            span.start_time_unix_nano = 1_000_000_000
            span.end_time_unix_nano = 2_000_000_000

            attr = span.attributes.add()
            attr.key = "test_key"
            attr.value.string_value = "test_value"

            resource = Resource()
            resource_attr = resource.attributes.add()
            resource_attr.key = "service.name"
            resource_attr.value.string_value = "test_service"

            scope_spans = ScopeSpans()
            scope_spans.scope.CopyFrom(InstrumentationScope(name='test_scope'))
            scope_spans.spans.append(span)

            resource_spans = ResourceSpans()
            resource_spans.resource.CopyFrom(resource)
            resource_spans.scope_spans.append(scope_spans)

            request = ExportTraceServiceRequest()
            request.resource_spans.append(resource_spans)

            response = stub.Export(request)
            self.assertIsNotNone(response)

            # Give the servicer thread a moment to record the span.
            deadline = time.time() + 2.0
            while time.time() < deadline:
                if graphsignal.sdk.sdk().span_store().has_unexported():
                    break
                time.sleep(0.01)

            exported = graphsignal.sdk.sdk().span_store().export()
            self.assertEqual(len(exported), 1)
            recorded = exported[0]
            # service.name was used as the engine prefix.
            self.assertEqual(recorded.name, "test_service.test_span")
            self.assertEqual(recorded.start_ts, 1_000_000_000)
            self.assertEqual(recorded.end_ts, 2_000_000_000)
            self.assertEqual(find_attribute(recorded, 'test_key'), 'test_value')
            # Resource attribute also flows through to span attributes.
            self.assertEqual(find_attribute(recorded, 'service.name'), 'test_service')
            self.assertEqual(find_counter(recorded, 'span.duration'),
                             float(2_000_000_000 - 1_000_000_000))
        finally:
            collector.shutdown()

    def test_servicer_drops_invalid_span(self):
        try:
            import grpc
            from opentelemetry.proto.collector.trace.v1.trace_service_pb2_grpc import TraceServiceStub
            from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
            from opentelemetry.proto.trace.v1.trace_pb2 import ResourceSpans, ScopeSpans, Span as OtelSpan
        except ImportError:
            self.skipTest("gRPC / OTLP protobufs not available")

        graphsignal.sdk.sdk().config_loader()._options['traces_per_sec'] = '1000'

        collector = OTELCollector(port=OTELCollector.find_port())
        collector.setup()
        try:
            endpoint = collector.get_endpoint()
            if not endpoint:
                self.skipTest("OTEL collector did not start")

            port = endpoint.split(':')[-1]
            channel = grpc.insecure_channel(f'localhost:{port}')
            stub = TraceServiceStub(channel)

            # Missing trace_id / span_id → dropped.
            span = OtelSpan()
            span.name = "no_ids"
            span.start_time_unix_nano = 1_000_000_000
            span.end_time_unix_nano = 2_000_000_000

            scope_spans = ScopeSpans()
            scope_spans.spans.append(span)
            resource_spans = ResourceSpans()
            resource_spans.scope_spans.append(scope_spans)
            request = ExportTraceServiceRequest()
            request.resource_spans.append(resource_spans)

            response = stub.Export(request)
            self.assertIsNotNone(response)

            time.sleep(0.1)
            self.assertFalse(graphsignal.sdk.sdk().span_store().has_unexported())
        finally:
            collector.shutdown()

    def test_profiles_span_when_export_dropped_by_sampling(self):
        sdk = graphsignal.sdk.sdk()
        sdk.span_store()._initial_trace_exports_remaining = 0
        sdk.config_loader()._options['traces_per_sec'] = '0'

        profiler = sdk._span_profiler
        if profiler is not None:
            profiler.shutdown()

        mock_profiler = MagicMock()
        sdk._span_profiler = mock_profiler

        span = MagicMock()
        span.trace_id = _make_trace_id_bytes(0xA2)
        span.span_id = _make_span_id_bytes(0xB2)
        span.parent_span_id = b''
        span.name = 'decode_forward'
        span.start_time_unix_nano = 1_000
        span.end_time_unix_nano = 2_000
        span.attributes = []
        span.events = []

        recorded = OTELCollectorServicer._record(
            sdk, span, {'service.name': 'sglang'})

        self.assertFalse(recorded)
        mock_profiler.record_span.assert_called_once_with(
            op_name='sglang.decode_forward',
            category='engine.otel',
            start_ns=1_000,
            end_ns=2_000,
            token_stats=None,
        )
        self.assertFalse(sdk.span_store().has_unexported())

    def test_servicer_passes_token_stats_to_profiler(self):
        sdk = graphsignal.sdk.sdk()
        sdk.span_store()._initial_trace_exports_remaining = 0
        sdk.config_loader()._options['traces_per_sec'] = '0'

        profiler = sdk._span_profiler
        if profiler is not None:
            profiler.shutdown()

        mock_profiler = MagicMock()
        sdk._span_profiler = mock_profiler

        span = MagicMock()
        span.trace_id = _make_trace_id_bytes(0xA3)
        span.span_id = _make_span_id_bytes(0xB3)
        span.parent_span_id = b''
        span.name = 'llm_request'
        span.start_time_unix_nano = 1_000_000_000
        span.end_time_unix_nano = 1_100_000_000
        span.attributes = [
            _make_otlp_attr('gen_ai.usage.prompt_tokens', int_value=100),
            _make_otlp_attr('gen_ai.usage.completion_tokens', int_value=60),
            _make_otlp_attr('gen_ai.latency.time_to_first_token', double_value=0.04),
        ]
        span.events = []

        OTELCollectorServicer._record(
            sdk, span, {'service.name': 'trtllm-server'})

        mock_profiler.record_span.assert_called_once()
        _, kwargs = mock_profiler.record_span.call_args
        self.assertEqual(kwargs['op_name'], 'trtllm-server.llm_request')
        self.assertEqual(kwargs['category'], 'engine.otel')
        self.assertEqual(kwargs['start_ns'], 1_000_000_000)
        self.assertEqual(kwargs['end_ns'], 1_100_000_000)
        token_stats = kwargs['token_stats']
        self.assertIsNotNone(token_stats)
        self.assertEqual(token_stats.input_tokens, 100)
        self.assertEqual(token_stats.output_tokens, 60)
        self.assertEqual(token_stats.phase_latency_ns, 40_000_000)


if __name__ == '__main__':
    unittest.main()
