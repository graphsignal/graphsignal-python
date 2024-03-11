import logging


logger = logging.getLogger('graphsignal')


def find_tag(proto, key):
    for tag in proto.tags:
        if tag.key == key:
            return tag.value
    return None


def find_payload(proto, name):
    for payload in proto.payloads:
        if payload.name == name:
            return payload
    return None

def find_usage(proto, payload_name, usage_name):
    for usage_counter in proto.usage:
        if (not payload_name or usage_counter.payload_name == payload_name) and usage_counter.name == usage_name:
            return usage_counter.value
    return None

