#!/bin/bash

set -e 

source venv/bin/activate
pip install --upgrade pip
pip install --upgrade setuptools
pip install --upgrade torch
if [ `uname -m` = "aarch64" ]; then
    export PIP_EXTRA_INDEX_URL=https://snapshots.linaro.org/ldcg/python-cache/ 
    pip install --upgrade tensorflow-aarch64
else
    pip install --upgrade tensorflow
fi
pip install --upgrade openai
pip install --upgrade tiktoken
pip install --upgrade langchain
pip install --upgrade transformers[agents]
pip install --upgrade yappi
pip install --upgrade banana-dev
pip install --upgrade chromadb
pip install --upgrade llama_index
pip install --upgrade nltk
deactivate
