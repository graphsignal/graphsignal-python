#!/bin/bash

set -e

rm -rf graphsignal/client
cp -r ../platform/clients/generated/python_client/graphsignal/client graphsignal
