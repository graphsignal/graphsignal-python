#!/bin/bash

source venv/bin/activate
RUN_VLLM_TESTS=1 python -m unittest test.recorders.test_vllm_recorder.VLLMRecorderTest.test_llm_generate -v
RUN_VLLM_TESTS=1 python -m unittest test.recorders.test_vllm_recorder.VLLMRecorderTest.test_async_llm_generate -v
deactivate