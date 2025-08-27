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
except ImportError as e:
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

    def export(self, spans):
        try:
            if self._export_callback:
                self._export_callback(spans)
        except Exception as e:
            logger.error(f"Error during OpenTelemetry export: {e}", exc_info=True)
        return


class OTELAdapter():
    def __init__(self, export_callback=None):
        self._provider = None
        self._exporter = None
        self._local_exporter = None
        self._export_callback = export_callback

    def setup(self):
        if not OTEL_AVAILABLE:
            logger.warning("OpenTelemetry not available, OTEL adapter setup skipped")
            return
            
        try:
            provider = trace.get_tracer_provider()
                        
            if isinstance(provider, TracerProvider):
                self._local_exporter = LocalSpanExporter(self._export_callback)
                try:
                    provider.add_span_processor(BatchSpanProcessor(self._local_exporter))
                except Exception as e:
                    logger.error(f"Failed to add span processor to existing provider: {e}", exc_info=True)
                    # Fall back to creating new provider
            else:
                logger.error("OpenTelemetry tracer provider not found, skipping setup")

            logger.info(f"OpenTelemetry tracer provider configured.")
        except Exception as e:
            logger.error(f"Failed to set up OpenTelemetry tracer provider: {e}", exc_info=True)

    def shutdown(self):
        try:
            if self._provider:
                self._provider.force_flush()
                logger.debug("OpenTelemetry tracer provider flushed")
        except Exception as e:
            logger.error(f"Error during OpenTelemetry shutdown: {e}", exc_info=True)
