import logging

import graphsignal
from graphsignal.profiling_step import ProfilingStep

logger = logging.getLogger('graphsignal')


def profile_step(ensure_profile=False):
    graphsignal._check_configured()

    return ProfilingStep(ensure_profile=ensure_profile)
