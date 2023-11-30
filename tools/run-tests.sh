#!/bin/bash

set -e

source venv/bin/activate
python -m unittest test/test_*.py
python -m unittest test/samplers/test_*.py
python -m unittest test/recorders/test_instrumentation.py
python -m unittest test/recorders/test_process_recorder.py
python -m unittest test/recorders/test_cprofile_recorder.py
python -m unittest test/recorders/test_kineto_recorder.py
python -m unittest test/recorders/test_yappi_recorder.py
python -m unittest test/recorders/test_nvml_recorder.py
python -m unittest test/recorders/test_pytorch_recorder.py
python -m unittest test/recorders/test_openai_recorder.py
python -m unittest test/recorders/test_langchain_recorder.py
python -m unittest test/recorders/test_llama_index_recorder.py
#python -m unittest test/recorders/test_huggingface_recorder.py
#python -m unittest test/recorders/test_banana_recorder.py
#python -m unittest test/recorders/test_chroma_recorder.py
python -m unittest test/callbacks/langchain/test_v2.py
python -m unittest test/callbacks/llama_index/test_v1.py
python -m unittest test/data/test_builtin_types.py
python -m unittest test/data/test_numpy_ndarray.py
python -m unittest test/data/test_tf_tensor.py
python -m unittest test/data/test_torch_tensor.py
deactivate
