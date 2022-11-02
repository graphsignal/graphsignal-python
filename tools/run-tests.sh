#!/bin/bash

set -e

source venv/bin/activate
python -m unittest graphsignal/*_test.py
python -m unittest graphsignal/usage/*_test.py
python -m unittest graphsignal/data/builtin_types_test.py
python -m unittest graphsignal/data/numpy_ndarray_test.py
python -m unittest graphsignal/data/tf_tensor_test.py
python -m unittest graphsignal/data/torch_tensor_test.py
python -m unittest graphsignal/data/missing_value_detector_test.py
python -m unittest graphsignal/profilers/python_test.py
python -m unittest graphsignal/profilers/tensorflow_test.py
python -m unittest graphsignal/profilers/pytorch_test.py
python -m unittest graphsignal/profilers/huggingface_subclass_test.py
python -m unittest graphsignal/profilers/huggingface_pipeline_test.py
python -m unittest graphsignal/profilers/jax_test.py
python -m unittest graphsignal/profilers/onnxruntime_test.py
python -m unittest graphsignal/callbacks/keras_test.py
python -m unittest graphsignal/callbacks/pytorch_lightning_test.py
python -m unittest graphsignal/callbacks/huggingface_test.py
deactivate
