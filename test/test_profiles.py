import unittest
import logging
import sys
import json
from unittest.mock import patch, Mock

import graphsignal
from graphsignal.profiles import EventAverages


logger = logging.getLogger('graphsignal')


class ProfilesTest(unittest.TestCase):
    def setUp(self):
        if len(logger.handlers) == 0:
            logger.addHandler(logging.StreamHandler(sys.stdout))
        graphsignal.configure(
            api_key='k1',
            tags={'deployment': 'd1', 'k1': 'v1'},
            debug_mode=True)
        graphsignal._tracer.hostname = 'h1'
        graphsignal._tracer.auto_export = False

    def tearDown(self):
        graphsignal.shutdown()

    def test_inc_counters(self):
        averages = EventAverages()

        # First event: add new counters
        averages.inc_counters('event1', {'latency': 10, 'errors': 1})
        self.assertEqual(averages.events['event1']['latency'], 10)
        self.assertEqual(averages.events['event1']['errors'], 1)
        self.assertEqual(averages.events['event1']['count'], 1)  # 'count' was not provided

        # Second call: increment existing counters
        averages.inc_counters('event1', {'latency': 5, 'errors': 2})
        self.assertEqual(averages.events['event1']['latency'], 15)
        self.assertEqual(averages.events['event1']['errors'], 3)
        self.assertEqual(averages.events['event1']['count'], 2)

        # Provide explicit 'count' in stats
        averages.inc_counters('event1', {'latency': 2, 'count': 5})
        self.assertEqual(averages.events['event1']['latency'], 17)
        self.assertEqual(averages.events['event1']['count'], 7)  # 2 (before) + 5 (explicit)

        # Add a new event with explicit 'count'
        averages.inc_counters('event2', {'latency': 100, 'count': 3})
        self.assertEqual(averages.events['event2']['latency'], 100)
        self.assertEqual(averages.events['event2']['count'], 3)


    def test_dumps_sorted_and_limited(self):
        averages = EventAverages()

        # Add mock events with different 'count' values
        averages.inc_counters('event1', {'count': 10})
        averages.inc_counters('event2', {'count': 5})
        averages.inc_counters('event3', {'count': 15})
        averages.inc_counters('event4', {'count': 1})

        # Dump only top 2 by 'count'
        result_json = averages.dumps(max_events=2, limit_by='count')
        self.assertIsInstance(result_json, str)

        result_dict = json.loads(result_json)
        self.assertEqual(len(result_dict), 2)

        # Verify correct top 2 by count
        self.assertIn('event3', result_dict)
        self.assertIn('event1', result_dict)
        self.assertNotIn('event2', result_dict)
        self.assertNotIn('event4', result_dict)