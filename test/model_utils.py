import logging


logger = logging.getLogger('graphsignal')


def find_tag(model, key):
    for tag in model.tags:
        if tag.key == key:
            return tag.value
    return None

def find_attribute(model, name):
    for attribute in model.attributes:
        if attribute.name == name:
            return attribute.value
    return None

def find_counter(model, counter_name):
    for counter in model.counters:
        if counter.name == counter_name:
            return counter.value
    return None

def find_profile(model, name):
    for profile in model.profiles:
        if profile.name == name:
            return profile
    return None

def find_log_entry(store, text):
    for entry in store._logs:
        if entry.message and text in entry.message:
            return entry
        if entry.exception and text in entry.exception:
            return entry
