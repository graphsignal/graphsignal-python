#!/bin/bash

set -e

protoc --proto_path=. --python_out=. ./graphsignal/profilers/tensorflow_proto/*.proto

protoc --proto_path=../platform/common/proto --python_out=./graphsignal/proto ../platform/common/proto/profiles.proto
