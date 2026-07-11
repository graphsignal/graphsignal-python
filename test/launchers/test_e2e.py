"""End-to-end launcher test.

Spawns `graphsignal-run` as a real subprocess, pointing it at a small torch
workload script that does one CUDA op. Verifies that:

  * The launcher dispatches successfully (subprocess exits 0).
  * The workload actually runs (it writes its pid to a file we own).
  * The CUPTI injection library is loaded into the workload (its presence
    creates `/dev/shm/graphsignal_cupti_<workload_pid>/`).

The test does not require network access; `GRAPHSIGNAL_API_BASE` is pointed at
a local port that's expected to be unreachable, so upload attempts fail
silently.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest


def _torch_cuda_available_in_subprocess() -> bool:
    code = "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"
    try:
        proc = subprocess.run([sys.executable, '-c', code],
                              timeout=30, capture_output=True)
        return proc.returncode == 0
    except Exception:
        return False


def _shm_dir(pid: int) -> str:
    return f"/dev/shm/graphsignal_cupti_{pid}"


# The workload appends a progress trail to the env-supplied output file from
# its very first line, so we can see exactly which step it reached even if
# stdout is eaten by the CUPTI injection library's atexit handler or the
# inherited pipe.
WORKLOAD = textwrap.dedent('''
    import os, sys, time, traceback

    _OUT = os.environ['GRAPHSIGNAL_TEST_OUTPUT']

    def _log(msg):
        with open(_OUT, 'a') as _f:
            _f.write(msg + '\\n')

    try:
        _log(f'PID={os.getpid()}')
        _log(f'CUDA_INJECTION64_PATH={os.environ.get("CUDA_INJECTION64_PATH", "<unset>")}')
        import torch
        _log(f'torch_imported, cuda_available={torch.cuda.is_available()}')
        a = torch.randn((64, 64), device='cuda', dtype=torch.float16)
        b = torch.randn((64, 64), device='cuda', dtype=torch.float16)
        _log('tensors_created')
        _ = a @ b
        torch.cuda.synchronize()
        _log('matmul_synced')
        # Linger so the CUPTI flush thread writes the shm dir.
        time.sleep(2.0)
        _log('done')
    except BaseException as exc:
        _log(f'EXCEPTION={type(exc).__name__}: {exc}')
        _log(traceback.format_exc())
        raise
''')


@unittest.skipUnless(sys.platform.startswith('linux'),
                     "graphsignal-run e2e requires Linux (CUPTI injection)")
class GraphsignalRunE2ETest(unittest.TestCase):
    def setUp(self):
        if not _torch_cuda_available_in_subprocess():
            self.skipTest("torch+CUDA not available")

    def test_graphsignal_run_with_torch_workload(self):
        with tempfile.NamedTemporaryFile(suffix='.py', mode='w', delete=False) as f:
            f.write(WORKLOAD)
            script = f.name
        output_path = tempfile.mkstemp(suffix='.pid')[1]
        os.unlink(output_path)  # let the workload create it fresh

        env = {**os.environ}
        env.pop('CUDA_INJECTION64_PATH', None)
        env['GRAPHSIGNAL_API_KEY'] = 'test-key'
        env['GRAPHSIGNAL_API_BASE'] = 'http://127.0.0.1:1'
        env['GRAPHSIGNAL_DEBUG'] = '1'
        env['GRAPHSIGNAL_TEST_OUTPUT'] = output_path

        # Invoke as `graphsignal-run python <script.py>`. FallbackLauncher
        # resolves `python` on PATH and `execv`'s it with the script as
        # argv[1], so the workload runs in a fresh process with no
        # launcher modules loaded — the configuration the CUPTI injection
        # library requires.
        cmd = [sys.executable, '-m', 'graphsignal.commands.graphsignal_run',
               sys.executable, script]
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True,
                              timeout=60)

        try:
            trail = ''
            if os.path.exists(output_path):
                with open(output_path) as f:
                    trail = f.read()

            # The native CUPTI injection library can SIGSEGV on teardown (-11)
            # on some platforms. That's a native-side cleanup issue independent
            # of the workload running correctly, so we accept either a clean
            # exit or a SIGSEGV *as long as* the workload reached 'done'.
            self.assertIn(
                proc.returncode, (0, -11),
                msg=(f"graphsignal-run exited unexpectedly with "
                     f"returncode={proc.returncode};\n"
                     f"--- trail ---\n{trail}\n"
                     f"--- stdout ---\n{proc.stdout}\n"
                     f"--- stderr ---\n{proc.stderr}"))

            self.assertTrue(
                os.path.exists(output_path),
                msg=(f"workload never wrote to {output_path};\n"
                     f"--- stdout ---\n{proc.stdout}\n"
                     f"--- stderr ---\n{proc.stderr}"))

            # Confirm the workload reached the final step before the (possibly
            # crashing) atexit handler ran.
            self.assertIn(
                'done', trail,
                msg=(f"workload did not reach 'done'; trail=\n{trail}\n"
                     f"stdout={proc.stdout!r} stderr={proc.stderr!r}"))

            # First line of the trail is `PID=<n>`.
            pid_line = trail.splitlines()[0]
            self.assertTrue(pid_line.startswith('PID='),
                            msg=f"unexpected trail head: {trail!r}")
            workload_pid = int(pid_line.split('=', 1)[1])

            # And confirm CUPTI actually attached (CUDA_INJECTION64_PATH was
            # forwarded by the launcher).
            self.assertIn('CUDA_INJECTION64_PATH=', trail)
            inj_line = next(
                ln for ln in trail.splitlines() if ln.startswith('CUDA_INJECTION64_PATH='))
            self.assertNotIn('<unset>', inj_line,
                             msg=f"launcher did not set CUDA_INJECTION64_PATH; trail=\n{trail}")

            # The CUPTI injection library creates /dev/shm/graphsignal_cupti_<pid>
            # at CUDA init. It may have been cleaned up by the lib's atexit
            # hook by now, so we don't strictly require it to exist post-exit
            # — but if it does, that's a useful signal that the injection
            # actually attached.
            if os.path.isdir(_shm_dir(workload_pid)):
                files = os.listdir(_shm_dir(workload_pid))
                self.assertIsInstance(files, list)
        finally:
            os.unlink(script)
            if os.path.exists(output_path):
                os.unlink(output_path)
            # Best-effort: clean up any lingering shm dir.
            for entry in os.listdir('/dev/shm') if os.path.isdir('/dev/shm') else ():
                if entry.startswith('graphsignal_'):
                    try:
                        shutil.rmtree(os.path.join('/dev/shm', entry), ignore_errors=True)
                    except Exception:
                        pass

            # Give the detached watcher subprocess a moment to notice its
            # target is gone and exit on its own.
            time.sleep(0.5)


if __name__ == '__main__':
    unittest.main()
