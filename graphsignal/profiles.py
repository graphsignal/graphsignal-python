from typing import Union, Any, Optional, Dict
import logging
import json

logger = logging.getLogger('graphsignal')


class EventAverages():
    __slots__ = [
        'events'
    ]

    def __init__(self):
        self.events = {}

    def is_empty(self) -> bool:
        return len(self.events) == 0

    def inc_counters(self, event_name, stats):
        if event_name not in self.events:
            event_counters = self.events[event_name] = dict()
        else:
            event_counters = self.events[event_name]

        for key, value in stats.items():
            if key not in event_counters:
                event_counters[key] = value
            else:
                event_counters[key] += value

        if 'count' not in stats:
            if 'count' not in event_counters:
                event_counters['count'] = 1
            else:
                event_counters['count'] += 1

    def dumps(self, max_events: int = 250, limit_by: str = 'count') -> str:
        # sort by limit_by key of event counters in descending order and return first max_events
        sorted_events = sorted(self.events.items(), key=lambda x: x[1].get(limit_by, 0), reverse=True)
        limited_events = dict(sorted_events[:max_events])
        
        return json.dumps(limited_events)

    def clear(self):
        self.events.clear()