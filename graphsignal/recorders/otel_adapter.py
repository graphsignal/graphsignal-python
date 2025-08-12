import os
import logging
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

from graphsignal.recorders.base_recorder import BaseRecorder

logger = logging.getLogger('graphsignal')


class LocalSpanExporter(SpanExporter):
    def export(self, spans):
        try:
            self._export_callback(spans)
        except Exception as e:
            logger.error(f"Error during OpenTelemetry export: {e}", exc_info=True)
        return SpanExporter.ResultCode.SUCCESS


class OTELAdapter():
    def __init__(self, service_name='service'):
        self._service_name = service_name
        self._provider = None
        self._exporter = None
        self._local_exporter = None
        self._export_callback = None

    def setup(self, export_callback):
        self._export_callback = export_callback
        try:
            provider = trace.get_tracer_provider()
            if isinstance(provider, TracerProvider):
                self._local_exporter = LocalSpanExporter()
                provider.add_span_processor(BatchSpanProcessor(self._local_exporter))
            else:                            
                service_name = os.environ.get("OTEL_SERVICE_NAME", self._service_name)
                resource = Resource.create({"service.name": service_name})
                self._provider = TracerProvider(resource=resource)
                self._exporter = OTLPSpanExporter()
                self._local_exporter = LocalSpanExporter()
                self._provider.add_span_processor(BatchSpanProcessor(self._exporter))
                self._provider.add_span_processor(BatchSpanProcessor(self._local_exporter))
                trace.set_tracer_provider(self._provider)

            logger.info(f"OpenTelemetry tracer provider configured for service: {service_name}")
            
        except Exception as e:
            logger.error(f"Failed to set up OpenTelemetry tracer provider: {e}", exc_info=True)

    def shutdown(self):
        try:
            if self._provider:
                self._provider.force_flush()
                logger.debug("OpenTelemetry tracer provider flushed")
        except Exception as e:
            logger.error(f"Error during OpenTelemetry shutdown: {e}", exc_info=True)
