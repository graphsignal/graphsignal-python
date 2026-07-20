"""Shared test helpers for per-launcher test files.

`LaunchFixture` mocks the side effects that every launcher's `launch()` method
triggers (OTEL port discovery, CUPTI env setup, target resolution, and the
final `launch_supervised` handoff) so a test can assert on the launcher's argv
mutations and ordering without touching the OS.
"""

from unittest.mock import patch


class LaunchFixture:
    """Shared mocks for launch() smoke tests across launchers.

    Pass the launcher module (e.g. ``graphsignal.launchers.vllm_launcher``)
    so the patches land on the symbols actually referenced by that module
    (each launcher imports `launch_supervised` / `_resolve` by name).
    """

    def __init__(self, module):
        self.module = module
        self.find_port = patch.object(module.OTELCollector, 'find_port', return_value=4242)
        self.setup_env = patch.object(module.CuptiProfiler, 'setup_env_vars', return_value=True)
        self.launch_supervised = patch.object(module, 'launch_supervised')
        self.resolve = patch.object(module, '_resolve', return_value='/abs/exec')

    def __enter__(self):
        self.find_port_m = self.find_port.start()
        self.setup_env_m = self.setup_env.start()
        self.launch_supervised_m = self.launch_supervised.start()
        self.resolve_m = self.resolve.start()
        return self

    def __exit__(self, *exc):
        self.resolve.stop()
        self.launch_supervised.stop()
        self.setup_env.stop()
        self.find_port.stop()

    @property
    def launched_argv(self):
        """The argv handed to launch_supervised (resolved executable first)."""
        return self.launch_supervised_m.call_args[0][0]
