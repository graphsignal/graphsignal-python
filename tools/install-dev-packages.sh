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
pip install tensorflow
pip install openai
pip install tiktoken
pip install langchain
pip install yappi
pip install banana-dev
pip install chromadb
deactivate
