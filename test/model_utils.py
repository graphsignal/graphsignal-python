import logging


logger = logging.getLogger('graphsignal')


def find_tag(model, key):
    for tag in model.tags:
        if tag.key == key:
            return tag.value
    return None


def find_payload(model, name):
    for payload in model.payloads:
        if payload.name == name:
            return payload
    return None

def find_usage(model, usage_name):
    for usage_counter in model.usage:
        if usage_counter.name == usage_name:
            return usage_counter.value
    return None

