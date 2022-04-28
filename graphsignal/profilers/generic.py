import logging

import graphsignal
from graphsignal.profiling_step import ProfilingStep

logger = logging.getLogger('graphsignal')


def profile_step(effective_batch_size=None, ensure_profile=False):
    graphsignal._check_configured()

    return ProfilingStep(
        effective_batch_size=effective_batch_size,
        ensure_profile=ensure_profile)
