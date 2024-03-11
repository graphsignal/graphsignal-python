#!/bin/bash

set -e 

source venv/bin/activate
pip install --upgrade pip
pip install --upgrade setuptools
pip install --upgrade torch
pip install --upgrade openai
pip install --upgrade tiktoken
pip install --upgrade langchain
pip install --upgrade langchainhub
pip install --upgrade llama_index
pip install --upgrade llama-index-llms-langchain
pip install --upgrade nltk
deactivate
