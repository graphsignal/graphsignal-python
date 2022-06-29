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
pip install tensorflow_datasets
pip install torch
pip install pytorch_lightning
pip install transformers
pip install datasets
pip install sklearn
pip install xgboost
if [ `uname -m` = "x86_64" ]; then
    pip install torch==1.11.0+cu115 -f https://download.pytorch.org/whl/torch_stable.html
    pip install jax[cuda] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
fi
deactivate
