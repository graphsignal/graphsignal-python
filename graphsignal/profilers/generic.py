from typing import Optional
import logging

import graphsignal
from graphsignal.profiling_step import ProfilingStep

logger = logging.getLogger('graphsignal')


def profile_step(
        phase_name: Optional[str] = None,
        effective_batch_size: Optional[int] = None,
        ensure_profile: Optional[bool] = False) -> ProfilingStep:
    graphsignal._check_configured()

    return ProfilingStep(
        phase_name=phase_name,
        effective_batch_size=effective_batch_size,
        ensure_profile=ensure_profile)
