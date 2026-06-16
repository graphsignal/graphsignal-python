import argparse
import logging
import signal as signal_module
import sys

import graphsignal.sdk as gsdk

log = logging.getLogger(__name__)

USAGE = """
Watch a target process (and its descendants) and report profiling data.

Usage:
  graphsignal-watch --pid PID [--otel-collector-port PORT] [--metrics-port PORT]
"""


def main():
    parser = argparse.ArgumentParser(
        prog='graphsignal-watch',
        description='Watch a target process with the Graphsignal profiler',
        usage=USAGE.strip(),
    )
    parser.add_argument('--pid', type=int, required=True,
                        help='Target process PID to watch')
    parser.add_argument('--otel-collector-port', type=int, default=None,
                        help='Port for the local OTLP/gRPC collector')
    parser.add_argument('--metrics-port', type=int, default=None,
                        help='Port to scrape the Prometheus /metrics endpoint on')
    args = parser.parse_args()

    try:
        gsdk.configure(
            target_pid=args.pid,
            otel_collector_port=args.otel_collector_port,
            metrics_port=args.metrics_port,
        )
    except Exception as exc:
        log.error('graphsignal-watch: profiler failed to configure: %s', exc, exc_info=True)
        sys.exit(1)

    sdk = gsdk.sdk()
    terminated = sdk.target_terminated_event()

    def _signal_handler(signum, frame):
        log.debug('graphsignal-watch: received signal %s', signum)
        terminated.set()

    for sig in (signal_module.SIGINT, signal_module.SIGTERM):
        try:
            signal_module.signal(sig, _signal_handler)
        except (ValueError, OSError):
            pass

    # Block until the target terminates (or we receive a signal).
    terminated.wait()

    gsdk.shutdown()
    sys.exit(0)


if __name__ == '__main__':
    main()
