import logging
import os
import platform
import re
import sys
from typing import Optional

logger = logging.getLogger("graphsignal")


class CuptiProfiler:
    """Holds CUPTI environment setup that must run before the target's CUDA initialization.

    Launchers call setup_env_vars() before exec'ing the target so the injection library is preloaded.
    """

    @staticmethod
    def setup_env_vars(cuda_graph_trace: Optional[str] = None) -> bool:
        if cuda_graph_trace is not None:
            mode = cuda_graph_trace.strip().lower()
            if mode not in ('graph', 'node'):
                raise ValueError(
                    "cuda_graph_trace must be 'graph' or 'node', got %r" % cuda_graph_trace)
            os.environ['GRAPHSIGNAL_CUDA_GRAPH_TRACE'] = mode

        if not sys.platform.startswith("linux"):
            logger.debug("CUPTI not supported on this platform")
            return False

        cuda_major = _detect_cuda_major()
        if not cuda_major:
            logger.debug("CUDA not available, skipping CUPTI env setup")
            return False

        if not _ensure_cuda_injection64_path():
            logger.debug("CUPTI profiler shared library not found")
            return False

        if not _ensure_libcupti_ld_library_path(prefer_major=cuda_major):
            logger.warning(
                "libcupti.so not found (CUDA %s detected), skipping CUPTI env setup. "
                "Install Graphsignal Python package with CUPTI extras (graphsignal[cu12] or graphsignal[cu13]) "
                "or install a CUDA toolkit that includes CUPTI (e.g. libcupti-dev / cuda-toolkit-%s-x).",
                cuda_major, cuda_major,
            )
            return False

        logger.debug("CUPTI env setup complete (cuda %s)", cuda_major)
        return True


def _packaged_cupti_so_path() -> Optional[str]:
    try:
        from importlib import resources

        cuda_major = _detect_cuda_major()
        arch = _detect_arch_tag()

        if cuda_major is None:
            return None

        candidate = resources.files("graphsignal").joinpath(
            "_native", f"{arch}-cu{cuda_major}", "libgscuptiprof.so"
        )
        with resources.as_file(candidate) as fp:
            if fp.exists():
                return str(fp)
            else:
                logger.debug("CUPTI profiler shared library not found for path: %s", candidate)
    except Exception:
        pass

    return None


def _detect_arch_tag() -> str:
    m = platform.machine().lower()
    if m in ("aarch64", "arm64"):
        return "arm64"
    return "amd64"


def _detect_cuda_major() -> Optional[int]:
    import ctypes.util

    for var in ('CUDA_VERSION', 'CUDA_TOOLKIT_VERSION'):
        val = os.environ.get(var, '')
        if val:
            m = re.match(r'^(\d+)', val.strip())
            if m:
                return int(m.group(1))

    torch_mod = sys.modules.get('torch')
    if torch_mod is not None:
        try:
            s = getattr(torch_mod.version, 'cuda', None)
            if isinstance(s, str):
                m = re.match(r'^(\d+)', s.strip())
                if m:
                    return int(m.group(1))
        except Exception:
            pass

    try:
        name = ctypes.util.find_library('cudart')
        if name:
            m = re.search(r'\.so\.(\d+)', name)
            if m:
                return int(m.group(1))
    except Exception:
        pass

    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH') or '/usr/local/cuda'
    search_dirs = [
        os.path.join(cuda_home, 'lib64'),
        os.path.join(cuda_home, 'lib'),
        '/usr/lib/x86_64-linux-gnu',
        '/usr/lib/aarch64-linux-gnu',
        '/usr/local/lib',
    ]
    for major in (13, 12, 11):
        for d in search_dirs:
            if os.path.isfile(os.path.join(d, f'libcudart.so.{major}')):
                return major

    return None


def _ensure_cuda_injection64_path() -> Optional[str]:
    if not sys.platform.startswith("linux"):
        return None

    existing = os.getenv("CUDA_INJECTION64_PATH")
    if existing:
        return existing

    p = _packaged_cupti_so_path()
    if not p:
        return None

    os.environ["CUDA_INJECTION64_PATH"] = p
    return p


def _ensure_libcupti_ld_library_path(*, prefer_major: Optional[int] = None) -> bool:
    sonames: tuple
    if prefer_major == 12:
        sonames = ("libcupti.so.12",)
    elif prefer_major == 13:
        sonames = ("libcupti.so.13",)
    else:
        sonames = ("libcupti.so.13", "libcupti.so.12", "libcupti.so")

    def _has_libcupti(d: str) -> bool:
        if not os.path.isdir(d):
            return False
        if any(os.path.exists(os.path.join(d, s)) for s in sonames):
            return True
        return any(e.startswith("libcupti.so") for e in os.listdir(d))

    def _prepend(lib_dir: str) -> None:
        existing = os.environ.get("LD_LIBRARY_PATH", "")
        dirs = [d for d in existing.split(":") if d]
        if lib_dir not in dirs:
            os.environ["LD_LIBRARY_PATH"] = ":".join([lib_dir] + dirs)
            logger.debug("Added CUPTI lib dir to LD_LIBRARY_PATH: %s", lib_dir)

    import ctypes.util
    if ctypes.util.find_library("cupti"):
        return True

    try:
        import importlib.util as _iutil
        spec = _iutil.find_spec("nvidia.cuda_cupti")
        if spec and spec.submodule_search_locations:
            pkg_dir = next(iter(spec.submodule_search_locations), None)
            if pkg_dir:
                d = os.path.join(pkg_dir, "lib")
                if _has_libcupti(d):
                    _prepend(d)
                    return True
    except Exception as exc:
        logger.debug("Failed to locate nvidia.cuda_cupti package: %s", exc)

    try:
        import importlib.metadata as _meta
        dist_names = [f"nvidia-cuda-cupti-cu{prefer_major}"] if prefer_major else []
        dist_names.append("nvidia-cuda-cupti")
        for dist_name in dist_names:
            try:
                dist = _meta.distribution(dist_name)
                for f in dist.files or ():
                    if "libcupti.so" in str(f):
                        full_path = str(dist.locate_file(f))
                        if os.path.exists(full_path):
                            _prepend(os.path.dirname(full_path))
                            return True
            except _meta.PackageNotFoundError:
                continue
    except Exception as exc:
        logger.debug("Failed to locate CUPTI via importlib.metadata: %s", exc)

    cuda_home = os.getenv("CUDA_HOME") or os.getenv("CUDA_PATH") or "/usr/local/cuda"
    for base in (
        os.path.join(cuda_home, "extras", "CUPTI", "lib64"),
        os.path.join(cuda_home, "extras", "CUPTI", "lib"),
        os.path.join(cuda_home, "lib64"),
    ):
        if _has_libcupti(base):
            _prepend(base)
            return True

    return False
