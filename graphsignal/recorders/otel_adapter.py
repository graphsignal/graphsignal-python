import os
import logging

# Check OpenTelemetry imports
try:
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    OTEL_AVAILABLE = True
    print("[DEBUG] OpenTelemetry imports successful")
except ImportError as e:
    print(f"[DEBUG] OpenTelemetry import failed: {e}")
    OTEL_AVAILABLE = False
    # Create dummy classes for graceful degradation
    class TracerProvider:
        pass
    class SpanExporter:
        pass
    class OTLPSpanExporter:
        pass

from graphsignal.recorders.base_recorder import BaseRecorder

logger = logging.getLogger('graphsignal')


class LocalSpanExporter(SpanExporter):
    def __init__(self, export_callback=None):
        super().__init__()
        self._export_callback = export_callback
        print(f"[DEBUG] LocalSpanExporter initialized with callback: {export_callback}")

    def export(self, spans):
        print(f"[DEBUG] LocalSpanExporter.export called with {len(spans)} spans")
        try:
            if self._export_callback:
                print(f"[DEBUG] Calling export callback with spans: {spans}")
                self._export_callback(spans)
            else:
                print(f"[DEBUG] No export callback set, spans: {spans}")
        except Exception as e:
            print(f"[DEBUG] Error during OpenTelemetry export: {e}")
            logger.error(f"Error during OpenTelemetry export: {e}", exc_info=True)
        return


class OTELAdapter():
    def __init__(self, export_callback=None):
        self._provider = None
        self._exporter = None
        self._local_exporter = None
        self._export_callback = export_callback

    def setup(self):
        print(f"[DEBUG] OTELAdapter.setup called")
        
        if not OTEL_AVAILABLE:
            logger.warning("OpenTelemetry not available, OTEL adapter setup skipped")
            return
            
        try:
            print(f"[DEBUG] Getting current tracer provider")
            provider = trace.get_tracer_provider()
            print(f"[DEBUG] Current tracer provider: {provider}, type: {type(provider)}")
                        
            if isinstance(provider, TracerProvider):
                print(f"[DEBUG] Provider is already configured, adding local exporter")
                self._local_exporter = LocalSpanExporter(self._export_callback)
                try:
                    provider.add_span_processor(BatchSpanProcessor(self._local_exporter))
                    print(f"[DEBUG] Added local exporter to existing provider")
                except Exception as e:
                    print(f"[DEBUG] Failed to add span processor to existing provider: {e}")
                    logger.error(f"Failed to add span processor to existing provider: {e}", exc_info=True)
                    # Fall back to creating new provider
            else:
                logger.error("OpenTelemetry tracer provider not found, skipping setup")

            logger.info(f"OpenTelemetry tracer provider configured.")
        except Exception as e:
            print(f"[DEBUG] Failed to set up OpenTelemetry tracer provider: {e}")
            logger.error(f"Failed to set up OpenTelemetry tracer provider: {e}", exc_info=True)

    def shutdown(self):
        print(f"[DEBUG] OTELAdapter.shutdown called")
        try:
            if self._provider:
                print(f"[DEBUG] Flushing provider")
                self._provider.force_flush()
                print(f"[DEBUG] Provider flushed successfully")
                logger.debug("OpenTelemetry tracer provider flushed")
            else:
                print(f"[DEBUG] No provider to flush")
        except Exception as e:
            print(f"[DEBUG] Error during OpenTelemetry shutdown: {e}")
            logger.error(f"Error during OpenTelemetry shutdown: {e}", exc_info=True)
