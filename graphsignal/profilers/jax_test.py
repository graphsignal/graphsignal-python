import unittest
import logging
import sys
import os
import json
import time
from unittest.mock import patch, Mock
from google.protobuf.json_format import MessageToJson
import pprint

import graphsignal
from graphsignal.proto import profiles_pb2
from graphsignal.uploader import Uploader

logger = logging.getLogger('graphsignal')


class JaxProfilerTest(unittest.TestCase):
    def setUp(self):
        if len(logger.handlers) == 0:
            logger.addHandler(logging.StreamHandler(sys.stdout))
        graphsignal.configure(
            api_key='k1',
            workload_name='w1',
            debug_mode=True)

    def tearDown(self):
        graphsignal.shutdown()

    @patch.object(Uploader, 'upload_profile')
    def test_profile_step(self, mocked_upload_profile):
        try:
            import jax
            import jax.numpy as jnp
            from jax import grad, jit, vmap
            from jax import random
        except ImportError:
            logger.info('Not testing JAX profiler, package not found.')
            return

        from graphsignal.profilers.jax import profile_step

        with profile_step(phase_name='training', ensure_profile=True):
            key = random.PRNGKey(0)
            x = random.normal(key, (10,))
            size = 100
            x = random.normal(key, (size, size), dtype=jnp.float32)
            jnp.dot(x, x.T).block_until_ready()

        profile = mocked_upload_profile.call_args[0][0]

        #pp = pprint.PrettyPrinter()
        #pp.pprint(MessageToJson(profile))

        self.assertEqual(
            profile.frameworks[0].type,
            profiles_pb2.FrameworkInfo.FrameworkType.JAX_FRAMEWORK)

        self.assertEqual(profile.phase_name, 'training')
        self.assertEqual(profile.step_stats.step_count, 1)
        self.assertTrue(profile.step_stats.total_time_us > 0)

        test_op_stats = None
        for op_stats in profile.op_stats:
            if op_stats.op_name == 'Thunk':
                test_op_stats = op_stats
                break
        self.assertIsNotNone(test_op_stats)
        self.assertTrue(test_op_stats.count >= 1)
        if jax.device_count() > 0:
            self.assertTrue(test_op_stats.total_device_time_us >= 1)
            self.assertTrue(test_op_stats.self_device_time_us >= 1)
        else:
            self.assertTrue(test_op_stats.total_host_time_us >= 1)
            self.assertTrue(test_op_stats.self_host_time_us >= 1)

        self.assertNotEqual(profile.trace_data, b'')
