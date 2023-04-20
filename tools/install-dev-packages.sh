#!/bin/bash

set -e 

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install --upgrade setuptools
pip install wheel
pip install protobuf
pip install autopep8
pip install twine
pip install pandoc
if [ `uname -m` = "aarch64" ]; then
    export PIP_EXTRA_INDEX_URL=https://snapshots.linaro.org/ldcg/python-cache/ 
    pip install tensorflow-aarch64
else
    pip install tensorflow
fi
pip install torch
if [ -x "$(command -v nvcc -V)" ]; then
    pip install jax[cuda] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
fi
pip install onnxruntime
pip install deepspeed
pip install openai
pip install langchain
pip install yappi
pip install banana-dev
deactivate
