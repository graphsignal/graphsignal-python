#!/bin/bash

set -e

source venv/bin/activate
python -m unittest graphsignal/*_test.py
python -m unittest graphsignal/recorders/instrumentation_test.py
python -m unittest graphsignal/recorders/process_recorder_test.py
python -m unittest graphsignal/recorders/cprofile_recorder_test.py
python -m unittest graphsignal/recorders/kineto_recorder_test.py
python -m unittest graphsignal/recorders/yappi_recorder_test.py
python -m unittest graphsignal/recorders/nvml_recorder_test.py
python -m unittest graphsignal/recorders/pytorch_recorder_test.py
python -m unittest graphsignal/recorders/tensorflow_recorder_test.py
python -m unittest graphsignal/recorders/jax_recorder_test.py
python -m unittest graphsignal/recorders/onnxruntime_recorder_test.py
python -m unittest graphsignal/recorders/xgboost_recorder_test.py
python -m unittest graphsignal/recorders/deepspeed_recorder_test.py
python -m unittest graphsignal/recorders/openai_recorder_test.py
python -m unittest graphsignal/recorders/langchain_recorder_test.py
python -m unittest graphsignal/recorders/banana_recorder_test.py
python -m unittest graphsignal/data/builtin_types_test.py
python -m unittest graphsignal/data/numpy_ndarray_test.py
python -m unittest graphsignal/data/tf_tensor_test.py
python -m unittest graphsignal/data/torch_tensor_test.py
python -m unittest graphsignal/data/missing_value_detector_test.py
deactivate
