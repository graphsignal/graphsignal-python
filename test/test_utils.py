import logging


logger = logging.getLogger('graphsignal')


def find_tag(proto, key):
    for tag in proto.tags:
        if tag.key == key:
            return tag.value
    return None

def find_attribute(proto, name):
    for attribute in proto.attributes:
        if attribute.name == name:
            return attribute.value
    return None

def find_counter(proto, counter_name):
    for counter in proto.counters:
        if counter.name == counter_name:
            return counter.value
    return None

def find_log_entry(store, text):
    for batch in store._log_batches.values():
        for entry in batch.log_entries:
            if entry.message and text in entry.message:
                return entry
            if entry.exception and text in entry.exception:
                return entry
    return None

def find_last_datapoint(store, key):
    if key in store._metrics:
        return store._metrics[key].datapoints[-1]
    return None