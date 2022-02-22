#!/bin/bash

set -e

source venv/bin/activate
python -m unittest graphsignal/*_test.py
python -m unittest graphsignal/profilers/tensorflow_profiler_test.py
python -m unittest graphsignal/profilers/pytorch_profiler_test.py
python -m unittest graphsignal/callbacks/keras_test.py
python -m unittest graphsignal/callbacks/huggingface_test.py
deactivate