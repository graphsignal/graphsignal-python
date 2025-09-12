import os
import logging
import socket
import threading
import time
from typing import Optional, Callable

# Check OpenTelemetry imports
try:
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest, ExportTraceServiceResponse
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2_grpc import TraceServiceServicer, add_TraceServiceServicer_to_server
    import grpc
    OTEL_AVAILABLE = True
except ImportError as e:
    OTEL_AVAILABLE = False
    # Create dummy classes for graceful degradation
    class TraceServiceServicer:
        pass

logger = logging.getLogger('graphsignal')


class OTELCollectorServicer(TraceServiceServicer):
    def __init__(self, export_callback: Optional[Callable] = None):
        self._export_callback = export_callback
    
    def Export(self, request: ExportTraceServiceRequest, context):
        try:
            if self._export_callback and request.resource_spans:
                # Convert OTLP spans to OpenTelemetry spans for processing
                for resource_span in request.resource_spans:
                    for scope_span in resource_span.scope_spans:
                        for otlp_span in scope_span.spans:
                            # Convert OTLP span to OpenTelemetry span format
                            otel_span = self._convert_otlp_to_otel_span(otlp_span, resource_span.resource)
                            if otel_span:
                                self._export_callback([otel_span])
        except Exception as e:
            logger.error(f"Error processing OTLP trace request: {e}", exc_info=True)
        
        # Return success response
        return ExportTraceServiceResponse()
    
    def _convert_otlp_to_otel_span(self, otlp_span, resource):
        try:
            # Create a mock OpenTelemetry span with the OTLP data
            class MockOTELSpan:
                def __init__(self, otlp_span, resource):
                    self.name = otlp_span.name
                    self.start_time = otlp_span.start_time_unix_nano
                    self.end_time = otlp_span.end_time_unix_nano
                    self.attributes = {}
                    
                    # Convert attributes
                    for attr in otlp_span.attributes:
                        key = attr.key
                        if attr.value.string_value:
                            value = attr.value.string_value
                        elif attr.value.int_value:
                            value = attr.value.int_value
                        elif attr.value.double_value:
                            value = attr.value.double_value
                        elif attr.value.bool_value:
                            value = attr.value.bool_value
                        else:
                            continue
                        self.attributes[key] = value
                    
                    # Add resource attributes
                    if resource and resource.attributes:
                        for attr in resource.attributes:
                            key = attr.key
                            if attr.value.string_value:
                                value = attr.value.string_value
                            elif attr.value.int_value:
                                value = attr.value.int_value
                            elif attr.value.double_value:
                                value = attr.value.double_value
                            elif attr.value.bool_value:
                                value = attr.value.bool_value
                            else:
                                continue
                            self.attributes[key] = value
                            
            return MockOTELSpan(otlp_span, resource)
        except Exception as e:
            logger.error(f"Error converting OTLP span: {e}", exc_info=True)
            return None


class OTELCollector:
    def __init__(self, export_callback: Optional[Callable] = None, port: Optional[int] = None):
        self._export_callback = export_callback
        self._port = port
        self._endpoint = None
        self._server = None
        self._server_thread = None
        
    def _find_free_port(self) -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('localhost', 0))
            s.listen(1)
            port = s.getsockname()[1]
        return port
    
    def setup(self):
        if not OTEL_AVAILABLE:
            logger.warning("OpenTelemetry gRPC not available, OTEL collector setup skipped")
            return
            
        try:
            if self._port is None:
                self._port = self._find_free_port()
            
            from concurrent.futures import ThreadPoolExecutor
            self._server = grpc.server(ThreadPoolExecutor(max_workers=4))
            
            servicer = OTELCollectorServicer(export_callback=self._export_callback)
            add_TraceServiceServicer_to_server(servicer, self._server)
            
            listen_addr = f'localhost:{self._port}'
            self._server.add_insecure_port(listen_addr)
            self._server.start()
            
            self._endpoint = f'grpc://localhost:{self._port}'
            
            logger.info(f"OTEL collector started on {self._endpoint}")
            if self._export_callback:
                logger.debug("OTEL collector - spans will be processed via export callback")
            else:
                logger.debug("OTEL collector - spans will be accepted but not processed (no callback)")
            
        except Exception as e:
            logger.error(f"Failed to set up OTEL collector: {e}", exc_info=True)
            self._server = None
            self._endpoint = None
    
    def get_endpoint(self) -> Optional[str]:
        return self._endpoint
    
    def get_port(self) -> Optional[int]:
        return self._port
    
    def shutdown(self):
        try:
            if self._server:
                self._server.stop(grace=2.0)
                logger.debug("OTEL collector stopped")
        except Exception as e:
            logger.error(f"Error during OTEL collector shutdown: {e}", exc_info=True)
        finally:
            self._server = None
            self._endpoint = None
