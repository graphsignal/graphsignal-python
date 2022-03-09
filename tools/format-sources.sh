#!/bin/bash

source venv/bin/activate
autopep8 --verbose --in-place --aggressive --recursive --exclude='*_pb2.py' --exclude='pynvml.py' graphsignal
deactivate