import unittest
import sys
import time
import logging
import threading
from unittest.mock import Mock

import graphsignal
from graphsignal.otel.otel_collector import OTELCollector

logger = logging.getLogger('graphsignal')


class OTELCollectorTest(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None
        if len(logger.handlers) == 0:
            logger.addHandler(logging.StreamHandler(sys.stdout))
        graphsignal.configure(
            api_key='k1',
            debug_mode=True)
        graphsignal._ticker.auto_tick = False

    def tearDown(self):
        graphsignal.shutdown()

    def test_collector_span_processing(self):
        received_spans = []
        
        def test_callback(spans):
            received_spans.extend(spans)
        
        collector = OTELCollector(export_callback=test_callback)
        collector.setup()
        
        try:
            # Try to import gRPC and create a channel
            import grpc
            from opentelemetry.proto.collector.trace.v1.trace_service_pb2_grpc import TraceServiceStub
            from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
            from opentelemetry.proto.trace.v1.trace_pb2 import ResourceSpans, ScopeSpans, Span, Resource, KeyValue, AnyValue
            from opentelemetry.proto.common.v1.common_pb2 import InstrumentationScope
            
            # Create a channel to the collector
            endpoint = collector.get_endpoint()
            if endpoint:
                # Extract port from grpc://localhost:PORT
                port = endpoint.split(':')[-1]
                channel = grpc.insecure_channel(f'localhost:{port}')
                stub = TraceServiceStub(channel)
                
                # Create a trace request with a span
                span = Span()
                span.name = "test_span"
                span.start_time_unix_nano = 1000000000  # 1 second in nanoseconds
                span.end_time_unix_nano = 2000000000    # 2 seconds in nanoseconds
                
                # Add an attribute
                attr = span.attributes.add()
                attr.key = "test_key"
                attr.value.string_value = "test_value"
                
                # Create resource
                resource = Resource()
                resource_attr = resource.attributes.add()
                resource_attr.key = "service.name"
                resource_attr.value.string_value = "test_service"
                
                # Create scope
                scope = InstrumentationScope()
                scope.name = "test_scope"
                
                # Create scope spans
                scope_spans = ScopeSpans()
                scope_spans.scope.CopyFrom(scope)
                scope_spans.spans.append(span)
                
                # Create resource spans
                resource_spans = ResourceSpans()
                resource_spans.resource.CopyFrom(resource)
                resource_spans.scope_spans.append(scope_spans)
                
                # Create request
                request = ExportTraceServiceRequest()
                request.resource_spans.append(resource_spans)
                
                # Send the request
                response = stub.Export(request)
                self.assertIsNotNone(response)
                
                # Give a moment for the callback to be called
                time.sleep(0.1)
                
                # Verify the callback was called with the span
                self.assertEqual(len(received_spans), 1)
                self.assertEqual(received_spans[0].name, "test_span")
                self.assertEqual(received_spans[0].start_time, 1000000000)
                self.assertEqual(received_spans[0].end_time, 2000000000)
                self.assertEqual(received_spans[0].attributes["test_key"], "test_value")
                self.assertEqual(received_spans[0].attributes["service.name"], "test_service")
                
        except ImportError:
            # Skip test if gRPC is not available
            self.skipTest("gRPC not available")
        finally:
            collector.shutdown()


    def test_collector_grpc_connection(self):
        collector = OTELCollector()
        collector.setup()
        
        try:
            # Try to import gRPC and create a channel
            import grpc
            from opentelemetry.proto.collector.trace.v1.trace_service_pb2_grpc import TraceServiceStub
            
            # Create a channel to the collector
            endpoint = collector.get_endpoint()
            if endpoint:
                # Extract port from grpc://localhost:PORT
                port = endpoint.split(':')[-1]
                channel = grpc.insecure_channel(f'localhost:{port}')
                stub = TraceServiceStub(channel)
                
                # Create a simple trace request
                from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
                request = ExportTraceServiceRequest()
                
                # This should not raise an exception (noop collector accepts everything)
                response = stub.Export(request)
                self.assertIsNotNone(response)
                
        except ImportError:
            # Skip test if gRPC is not available
            self.skipTest("gRPC not available")
        finally:
            collector.shutdown()


if __name__ == '__main__':
    unittest.main()
