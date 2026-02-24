"""
Bootstrapping code that is run when Graphsignal bootstrap directory is in PYTHONPATH.

This file is automatically loaded by Python when the bootstrap directory is in PYTHONPATH,
which allows automatic instrumentation of subprocesses.
"""
import os

try:
    if os.getenv('GRAPHSIGNAL_API_KEY'):
        import graphsignal
        graphsignal.configure()
except Exception:
    import logging
    log = logging.getLogger(__name__)
    log.warning("error configuring Graphsignal tracing via sitecustomize.py", exc_info=True)
