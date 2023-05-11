import logging
import os
import sys
import platform
import socket
import re

from graphsignal.proto import signals_pb2

logger = logging.getLogger('graphsignal')

version_regexp = re.compile(r'^(\d+)\.?(\d+)?\.?(\d+)?')


def parse_semver(semver_proto, version):
    version_match = version_regexp.match(str(version))
    if version_match is not None:
        groups = version_match.groups()
        if groups[0] is not None:
            semver_proto.major = int(groups[0])
        if groups[1] is not None:
            semver_proto.minor = int(groups[1])
        if groups[2] is not None:
            semver_proto.patch = int(groups[2])


def compare_semver(semver_proto, version):
    semver_int = semver_proto.major * 1e6 + semver_proto.minor * 1e3 + semver_proto.patch
    version_int = version[0] * 1e6 + version[1] * 1e3 + version[2]
    if semver_int < version_int:
        return -1
    if semver_int > version_int:
        return 1
    else:
        return 0


def add_library_param(library_info, name, value):
    param = library_info.params.add()
    param.name = name
    param.value = str(value)


def add_driver(node_info, name, version):
    driver = node_info.drivers.add()
    driver.name = name
    driver.version = version


def find_tag(proto, key):
    for tag in proto.tags:
        if tag.key == key:
            return tag.value
    return None


def find_param(proto, name):
    for param in proto.params:
        if param.name == name:
            return param.value
    return None


def find_data_count(proto, data_name, count_name):
    for data_stats in proto.data_profile:
        if data_stats.data_name == data_name:
            for data_count in data_stats.counts:
                if data_count.name == count_name:
                    return data_count.count
    return None


def find_data_sample(proto, data_name):
    for data_sample in proto.data_samples:
        if data_sample.data_name == data_name:
            return data_sample
    return None