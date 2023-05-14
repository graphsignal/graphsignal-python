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
pip install torch
if [ `uname -m` = "aarch64" ]; then
    export PIP_EXTRA_INDEX_URL=https://snapshots.linaro.org/ldcg/python-cache/ 
    pip install tensorflow-aarch64
else
    pip install tensorflow
fi
pip install openai
pip install tiktoken
pip install langchain
pip install transformers[agents]
pip install yappi
pip install banana-dev
pip install chromadb
deactivate
