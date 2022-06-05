import logging
import os
import sys
import platform
import socket
import re

from graphsignal.proto import profiles_pb2

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
    semver_int = semver_proto.major * 1e6 + \
        semver_proto.minor * 1e3 + semver_proto.patch
    version_int = version[0] * 1e6 + version[1] * 1e3 + version[2]
    if semver_int < version_int:
        return -1
    if semver_int > version_int:
        return 1
    else:
        return 0
