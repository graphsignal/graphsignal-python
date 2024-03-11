#!/bin/bash

set -e

source venv/bin/activate
python -m unittest test/test_*.py
python -m unittest test/recorders/test_instrumentation.py
python -m unittest test/recorders/test_process_recorder.py
python -m unittest test/recorders/test_nvml_recorder.py
python -m unittest test/recorders/test_openai_recorder.py
python -m unittest test/recorders/test_langchain_recorder.py
python -m unittest test/recorders/test_llama_index_recorder.py
python -m unittest test/callbacks/langchain/test_v2.py
python -m unittest test/callbacks/llama_index/test_v2.py
deactivate
