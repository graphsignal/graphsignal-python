#!/bin/bash

set -e

protoc --proto_path=../platform/common/proto --python_out=./graphsignal/proto ../platform/common/proto/signals.proto
