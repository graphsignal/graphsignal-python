#!/bin/bash

set -e 

python3 -m venv venv
source venv/bin/activate
pip install numpy
pip install pandas
pip install scipy
pip install scikit-learn
pip install autopep8
pip install twine
pip install pandoc
deactivate