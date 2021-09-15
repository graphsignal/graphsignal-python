#!/bin/bash

set -e

protoc -I=../platform/common/proto --python_out=./graphsignal ../platform/common/proto/metrics.proto
