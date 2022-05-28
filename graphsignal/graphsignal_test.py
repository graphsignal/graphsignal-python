import unittest
import logging
import sys
import os
from unittest.mock import patch, Mock

import graphsignal

logger = logging.getLogger('graphsignal')


class GraphsignalTest(unittest.TestCase):
    def setUp(self):
        if len(logger.handlers) == 0:
            logger.addHandler(logging.StreamHandler(sys.stdout))
        graphsignal.configure(
            api_key='k1',
            workload_name='w1',
            run_id='r1',
            node_rank=1,
            local_rank=2,
            global_rank=3,
            debug_mode=True)

    def tearDown(self):
        graphsignal.shutdown()

    def test_check_and_set_arg_test(self):
        arg1 = graphsignal._check_and_set_arg('arg1', 'val1', is_str=True, required=True)
        self.assertEqual(arg1, 'val1')

        arg2 = graphsignal._check_and_set_arg('arg2', 1, is_int=True, required=True)
        self.assertEqual(arg2, 1)

        arg3 = graphsignal._check_and_set_arg('arg3', None, is_int=True, required=False)
        self.assertEqual(arg3, None)

        os.environ['GRAPHSIGNAL_ARG4'] = '2'
        arg4 = graphsignal._check_and_set_arg('arg4', None, is_int=True, required=False)
        self.assertEqual(arg4, 2)

        with self.assertRaises(ValueError):
            arg5 = graphsignal._check_and_set_arg('arg5', None, is_str=True, required=True)

        os.environ['GRAPHSIGNAL_ARG6'] = '10'
        arg6 = graphsignal._check_and_set_arg('arg6', None, is_int=True, required=True)
        self.assertEqual(arg6, 10)

        os.environ['GRAPHSIGNAL_ARG7'] = 'str'
        with self.assertRaises(ValueError):
            arg7 = graphsignal._check_and_set_arg('arg7', None, is_int=True, required=True)


    def test_configure(self):
        self.assertTrue(graphsignal._agent.start_ms > 0)
        self.assertIsNotNone(graphsignal._agent.worker_id)
        self.assertEqual(graphsignal._agent.run_id, '5573e39b6600')
        self.assertEqual(graphsignal._agent.api_key, 'k1')
        self.assertEqual(graphsignal._agent.workload_name, 'w1')
        self.assertEqual(graphsignal._agent.node_rank, 1)
        self.assertEqual(graphsignal._agent.local_rank, 2)
        self.assertEqual(graphsignal._agent.global_rank, 3)
        self.assertEqual(graphsignal._agent.debug_mode, True)
