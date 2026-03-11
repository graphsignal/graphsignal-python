"""
Bootstrapping code loaded at interpreter startup for automatic instrumentation.

When graphsignal is installed via pip, graphsignal.pth adds this directory to
sys.path, so Python loads sitecustomize.py in every process (including vLLM/Ray
worker processes). No PYTHONPATH or other env setup is required.

If GRAPHSIGNAL_API_KEY is set, Graphsignal is configured automatically so that
subprocesses and workers get instrumentation without user code.
"""
import os

try:
    if os.getenv("GRAPHSIGNAL_API_KEY"):
        import graphsignal
        graphsignal.configure()
except Exception:
    import logging
    log = logging.getLogger(__name__)
    log.warning("error configuring Graphsignal tracing via sitecustomize.py", exc_info=True)
