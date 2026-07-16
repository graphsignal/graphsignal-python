import unittest
import logging
import os
import random
import sys
import time

import graphsignal
import graphsignal.sdk
from graphsignal.recorders.process_recorder import ProcessRecorder
from test.test_utils import find_last_datapoint

logger = logging.getLogger('graphsignal')

mem = []


class ProcessRecorderTest(unittest.TestCase):
    def setUp(self):
        if len(logger.handlers) == 0:
            logger.addHandler(logging.StreamHandler(sys.stdout))
        graphsignal.sdk.configure(
            api_key='k1',
            debug_mode=True)
        graphsignal.sdk.sdk()._auto_tick = False

    def tearDown(self):
        graphsignal.sdk.shutdown()

    def test_record(self):
        pid = os.getpid()
        recorder = ProcessRecorder(root_pid=pid, pid=pid,
                                   args='python -m unittest')
        recorder.setup()

        time.sleep(0.2)
        for _ in range(100000):
            random.random()
        global mem
        mem = [1] * 100000

        recorder.on_tick()

        store = graphsignal.sdk.sdk().metric_store()
        pid_tag = {'process.pid': str(pid)}

        key = store.metric_key('process.memory.usage', pid_tag)
        self.assertTrue(find_last_datapoint(store, key).gauge > 0)
        key = store.metric_key('process.memory.virtual', pid_tag)
        self.assertTrue(find_last_datapoint(store, key).gauge > 0)

        sdk = graphsignal.sdk.sdk()
        resource_store = sdk.resource_store()
        self.assertTrue(resource_store.has_unexported())
        resources = resource_store.export()

        process_resources = [r for r in resources if r.kind == 'process']
        self.assertEqual(len(process_resources), 1)
        process_resource = process_resources[0]
        process_tag_dict = {t.key: t.value for t in process_resource.tags}
        self.assertEqual(process_tag_dict.get('process.pid'), str(pid))
        # The root's own resource links to itself.
        self.assertEqual(process_tag_dict.get('process.root_pid'), str(pid))

        process_attr_names = [a.name for a in process_resource.attributes]
        self.assertIn('process.command_line', process_attr_names)
        # When watching self, runtime attrs are also reported.
        self.assertIn('runtime.name', process_attr_names)
        self.assertIn('runtime.version', process_attr_names)
