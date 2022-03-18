#!/bin/bash

source venv/bin/activate
autopep8 --verbose --max-line-length=119 --in-place --recursive --exclude='*_pb2.py' --exclude='pynvml.py' graphsignal
deactivate