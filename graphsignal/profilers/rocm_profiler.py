import logging
import os
import platform
import re
import sys
from typing import Optional

logger = logging.getLogger("graphsignal")


class RocmProfiler:
    """Holds ROCm environment setup that must run before the target's HIP/ROCm initialization.

    Launchers call setup_env_vars() before exec'ing the target so the
    rocprofiler-sdk tool library (libgsrocmprof.so) is registered via
    ROCP_TOOL_LIBRARIES.

    Unlike the CUPTI backend there is no pip wheel for the ROCm profiling stack:
    rocprofiler-sdk ships only as a system ROCm package (preinstalled in the
    target images, e.g. rocm/dev-ubuntu-*-complete or rocm/pytorch). If it is
    not found this method warns once and returns False (profiling stays off,
    the target runs unaffected), so it is safe to call unconditionally alongside
    CuptiProfiler.
    """

    @staticmethod
    def setup_env_vars() -> bool:
        if not sys.platform.startswith("linux"):
            logger.debug("ROCm not supported on this platform")
            return False

        rocm_major = _detect_rocm_major()
        if not rocm_major:
            logger.debug("ROCm not available, skipping ROCm env setup")
            return False

        tool_lib = _packaged_rocm_so_path(rocm_major)
        if not tool_lib:
            logger.warning(
                "ROCm profiler shared library (libgsrocmprof.so) not found for "
                "ROCm %s, skipping ROCm env setup.",
                rocm_major,
            )
            return False

        if not _ensure_librocprofiler_ld_library_path():
            logger.warning(
                "librocprofiler-sdk.so not found (ROCm %s detected), skipping ROCm env setup. "
                "Install the rocprofiler-sdk system package (it ships with ROCm) "
                "or use a ROCm image that includes it (e.g. rocm/dev-ubuntu-*-complete "
                "or rocm/pytorch). There is no pip package for the ROCm profiling stack.",
                rocm_major,
            )
            return False

        _ensure_rocp_tool_libraries(tool_lib)

        logger.debug("ROCm env setup complete (rocm %s)", rocm_major)
        return True


def _packaged_rocm_so_path(rocm_major: int) -> Optional[str]:
    try:
        from importlib import resources

        # ROCm is amd64 only.
        candidate = resources.files("graphsignal").joinpath(
            "_native", f"amd64-rocm{rocm_major}", "libgsrocmprof.so"
        )
        with resources.as_file(candidate) as fp:
            if fp.exists():
                return str(fp)
            else:
                logger.debug("ROCm profiler shared library not found for path: %s", candidate)
    except Exception:
        pass

    return None


def _detect_rocm_major() -> Optional[int]:
    for var in ('ROCM_VERSION', 'ROCM_TOOLKIT_VERSION'):
        val = os.environ.get(var, '')
        if val:
            m = re.match(r'^(\d+)', val.strip())
            if m:
                return int(m.group(1))

    torch_mod = sys.modules.get('torch')
    if torch_mod is not None:
        try:
            s = getattr(torch_mod.version, 'hip', None)
            if isinstance(s, str):
                m = re.match(r'^(\d+)', s.strip())
                if m:
                    return int(m.group(1))
        except Exception:
            pass

    # Infer from the ROCm install. Try, in order: the version file ROCm ships
    # (.info/version, e.g. "7.2.3-123"), then the resolved install-dir name
    # (e.g. /opt/rocm -> /opt/rocm-7.2.3). ROCM_PATH is often the unversioned
    # symlink "/opt/rocm", so resolve it before matching.
    #
    # NOTE: deliberately NOT using ctypes.util.find_library('rocprofiler-sdk'):
    # it returns the ABI soname (librocprofiler-sdk.so.1), whose "1" is the
    # library ABI version, not the ROCm major — it would wrongly report ROCm 1.
    for base in (os.environ.get('ROCM_PATH'), os.environ.get('ROCM_HOME'), '/opt/rocm'):
        if not base:
            continue
        major = _read_rocm_version_file(base)
        if major:
            return major
        for candidate in (base, os.path.realpath(base)):
            m = re.search(r'rocm-(\d+)', candidate)
            if m:
                return int(m.group(1))

    # Fall back to scanning /opt for a versioned ROCm install dir (e.g.
    # /opt/rocm-7.2.3). Pick the highest major found.
    try:
        majors = []
        for entry in os.listdir('/opt'):
            m = re.match(r'^rocm-(\d+)', entry)
            if m and os.path.isdir(os.path.join('/opt', entry)):
                majors.append(int(m.group(1)))
        if majors:
            return max(majors)
    except Exception:
        pass

    return None


def _read_rocm_version_file(base: str) -> Optional[int]:
    # ROCm ships a plain-text version file under <install>/.info/ (e.g.
    # "7.2.3-123"). Return its major, or None if unavailable/unparsable.
    for fname in ('.info/version', '.info/version-dev', '.info/version-utils'):
        try:
            with open(os.path.join(base, fname)) as f:
                content = f.read().strip()
        except Exception:
            continue
        m = re.match(r'^(\d+)', content)
        if m:
            return int(m.group(1))
    return None


def _ensure_rocp_tool_libraries(tool_lib: str) -> None:
    existing = os.environ.get("ROCP_TOOL_LIBRARIES", "")
    entries = [e for e in existing.split(":") if e]
    if tool_lib not in entries:
        os.environ["ROCP_TOOL_LIBRARIES"] = ":".join([tool_lib] + entries) if entries else tool_lib
        logger.debug("Added ROCm tool library to ROCP_TOOL_LIBRARIES: %s", tool_lib)


def _ensure_librocprofiler_ld_library_path() -> bool:
    def _has_lib(d: str) -> bool:
        if not os.path.isdir(d):
            return False
        try:
            return any(e.startswith("librocprofiler-sdk.so") for e in os.listdir(d))
        except Exception:
            return False

    def _prepend(lib_dir: str) -> None:
        existing = os.environ.get("LD_LIBRARY_PATH", "")
        dirs = [d for d in existing.split(":") if d]
        if lib_dir not in dirs:
            os.environ["LD_LIBRARY_PATH"] = ":".join([lib_dir] + dirs)
            logger.debug("Added rocprofiler-sdk lib dir to LD_LIBRARY_PATH: %s", lib_dir)

    import ctypes.util
    found = ctypes.util.find_library("rocprofiler-sdk")
    if found:
        # Already loadable; if it resolves to a concrete path, prepend its dir.
        d = os.path.dirname(found) if os.path.sep in found else ""
        if d and _has_lib(d):
            _prepend(d)
        return True

    rocm_path = os.environ.get("ROCM_PATH") or os.environ.get("ROCM_HOME") or "/opt/rocm"
    for base in (
        os.path.join(rocm_path, "lib"),
        os.path.join(rocm_path, "lib64"),
        "/opt/rocm/lib",
        "/opt/rocm/lib64",
    ):
        if _has_lib(base):
            _prepend(base)
            return True

    return False
