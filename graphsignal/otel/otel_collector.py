import logging
import socket
import threading
from typing import Any, Dict, Optional

from graphsignal.otel.span_op_name import stable_otel_op_name
from graphsignal.otel.span_token_stats import extract_span_token_stats

try:
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
        ExportTraceServiceRequest, ExportTraceServiceResponse)
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2_grpc import (
        TraceServiceServicer, add_TraceServiceServicer_to_server)
    import grpc
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    # Dummy base so the module is importable on systems without grpc.
    class TraceServiceServicer:  # type: ignore[no-redef]
        pass

# NOTE: `graphsignal.sdk` is imported lazily inside the methods that need it.
# Importing it at module load would loop: `graphsignal.sdk.sdk` imports
# `graphsignal.otel.otel_collector` for the OTELCollector class.

logger = logging.getLogger('graphsignal')


def _extract_any_value(any_value):
    value_type = any_value.WhichOneof('value')
    if value_type == 'string_value':
        return any_value.string_value
    if value_type == 'int_value':
        return any_value.int_value
    if value_type == 'double_value':
        return any_value.double_value
    if value_type == 'bool_value':
        return any_value.bool_value
    return None


def _collect_attributes(otlp_attributes) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for attr in otlp_attributes or ():
        value = _extract_any_value(attr.value)
        if value is None:
            continue
        out[attr.key] = value
    return out


def _otel_id_hex(raw_bytes) -> Optional[str]:
    if not raw_bytes:
        return None
    hex_value = raw_bytes.hex().strip().lower()
    return hex_value or None


def _has_parent(parent_hex: Optional[str]) -> bool:
    return bool(parent_hex) and parent_hex.strip('0') != ''


class OTELCollectorServicer(TraceServiceServicer):
    """Receives OTLP/gRPC trace exports and forwards each span to the SDK via
    `Sdk.record_span`. No callback indirection.
    """

    def __init__(self):
        self._total_received = 0
        self._total_recorded = 0
        self._logged_first = False

    def Export(self, request, context):  # type: ignore[override]
        # Local import to break the cycle with graphsignal.sdk.sdk.
        import graphsignal.sdk as gsdk
        try:
            if not gsdk.is_configured():
                # Late-arriving export after shutdown — drop quietly.
                logger.debug('OTEL export received but SDK not configured; dropping')
                return ExportTraceServiceResponse()
            sdk = gsdk.sdk()

            received = 0
            recorded = 0
            for resource_span in request.resource_spans:
                resource_attrs = _collect_attributes(
                    getattr(resource_span.resource, 'attributes', ()) if resource_span.resource else ())
                for scope_span in resource_span.scope_spans:
                    for otlp_span in scope_span.spans:
                        received += 1
                        if self._record(sdk, otlp_span, resource_attrs):
                            recorded += 1

            self._total_received += received
            self._total_recorded += recorded
            # One-time confirmation that the workload's exporter actually
            # reached the collector — distinguishes "not exporting" from
            # "exporting but dropped by sampling".
            if not self._logged_first and received:
                self._logged_first = True
                logger.info(
                    'OTEL collector received first export: %d span(s), %d recorded',
                    received, recorded)
            logger.debug(
                'OTEL export: received=%d recorded=%d (total received=%d recorded=%d)',
                received, recorded, self._total_received, self._total_recorded)
        except Exception as exc:
            logger.error('Error processing OTLP trace request: %s', exc, exc_info=True)

        return ExportTraceServiceResponse()

    @staticmethod
    def _record(sdk, otlp_span, resource_attrs: Dict[str, Any]) -> bool:
        trace_id = _otel_id_hex(otlp_span.trace_id)
        span_id = _otel_id_hex(otlp_span.span_id)
        if not trace_id or not span_id:
            return False
        parent_span_id = _otel_id_hex(otlp_span.parent_span_id)
        parent_for_record = parent_span_id if _has_parent(parent_span_id) else None

        attributes = _collect_attributes(otlp_span.attributes)
        for key, value in resource_attrs.items():
            attributes.setdefault(key, value)

        engine = attributes.get('service.name') if isinstance(attributes, dict) else None
        name = f'{engine}.{otlp_span.name}' if engine else otlp_span.name

        events = []
        for evt in otlp_span.events or ():
            events.append({
                'name': evt.name,
                'event_ts': evt.time_unix_nano,
                'attributes': _collect_attributes(evt.attributes),
            })

        start_ts = otlp_span.start_time_unix_nano
        end_ts = otlp_span.end_time_unix_nano

        span_profiler = sdk.span_profiler()
        if span_profiler is not None and start_ts > 0 and end_ts > 0:
            try:
                token_stats = extract_span_token_stats(attributes)
                span_profiler.record_span(
                    op_name=stable_otel_op_name(name, attributes),
                    category='engine.otel',
                    start_ns=start_ts,
                    end_ns=end_ts,
                    token_stats=token_stats)
            except Exception:
                pass

        span_proto = sdk.record_span(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_for_record,
            name=name,
            start_ts=start_ts,
            end_ts=end_ts,
            attributes=attributes,
            events=events,
        )
        return span_proto is not None


class OTELCollector:
    def __init__(self, port: int):
        self._port = int(port)
        self._endpoint: Optional[str] = None
        self._server = None
        self._lock = threading.Lock()

    @staticmethod
    def find_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('127.0.0.1', 0))
            s.listen(1)
            return s.getsockname()[1]

    def setup(self):
        if not OTEL_AVAILABLE:
            logger.warning("OpenTelemetry gRPC not available, OTEL collector setup skipped")
            return

        try:
            from concurrent.futures import ThreadPoolExecutor
            self._server = grpc.server(ThreadPoolExecutor(max_workers=4))

            servicer = OTELCollectorServicer()
            add_TraceServiceServicer_to_server(servicer, self._server)

            # Bind the explicit IPv4 loopback literal (not "localhost") so the
            # server and the workload's exporter can't disagree on IPv4 vs IPv6
            # — on hosts where "localhost" resolves to both ::1 and 127.0.0.1,
            # a name-based bind/dial can silently land on different families.
            listen_addr = f'127.0.0.1:{self._port}'
            bound_port = self._server.add_insecure_port(listen_addr)
            if bound_port == 0:
                # gRPC returns 0 when the bind fails (e.g. the port was taken
                # between find_port() in the launcher and this bind). Don't
                # pretend the collector is up — nothing would be listening.
                logger.error(
                    "OTEL collector failed to bind %s (port unavailable); "
                    "workload exports will go nowhere", listen_addr)
                self._server = None
                self._endpoint = None
                return
            self._server.start()

            self._endpoint = f'grpc://127.0.0.1:{bound_port}'
            logger.info("OTEL collector started on %s (bound_port=%d)",
                        self._endpoint, bound_port)
        except Exception as exc:
            logger.error("Failed to set up OTEL collector: %s", exc, exc_info=True)
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
        except Exception as exc:
            logger.error("Error during OTEL collector shutdown: %s", exc, exc_info=True)
        finally:
            self._server = None
            self._endpoint = None
