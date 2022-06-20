#!/bin/bash

set -e 

python3 -m venv venv
source venv/bin/activate
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
if [ `uname -m` = "aarch64" ]; then
    pip install torch
else
    #pip install torch
    pip install torch==1.11.0+cu115 -f https://download.pytorch.org/whl/torch_stable.html
fi
pip install pytorch_lightning
pip install torchmetrics
pip install torchvision
pip install transformers
pip install datasets
pip install sklearn
pip install xgboost
deactivate
