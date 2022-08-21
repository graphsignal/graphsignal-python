#!/bin/bash

set -e

source venv/bin/activate
python -m unittest graphsignal/*_test.py
python -m unittest graphsignal/usage/*_test.py
python -m unittest graphsignal/tracers/python_test.py
python -m unittest graphsignal/tracers/tensorflow_test.py
python -m unittest graphsignal/tracers/pytorch_test.py
python -m unittest graphsignal/tracers/keras_test.py
python -m unittest graphsignal/tracers/pytorch_lightning_test.py
python -m unittest graphsignal/tracers/huggingface_subclass_test.py
python -m unittest graphsignal/tracers/huggingface_pipeline_test.py
python -m unittest graphsignal/tracers/jax_test.py
python -m unittest graphsignal/tracers/onnxruntime_test.py
deactivate
