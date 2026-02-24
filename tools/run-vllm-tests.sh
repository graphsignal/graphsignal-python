#!/bin/bash

RUN_VLLM_TESTS=1 poetry run test test/recorders/test_vllm_recorder.py::VLLMRecorderTest::test_llm_generate
RUN_VLLM_TESTS=1 poetry run test test/recorders/test_vllm_recorder.py::VLLMRecorderTest::test_async_llm_generate
