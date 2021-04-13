#!/bin/bash

set -e

version=${1:-"3.8.3"}
venv="testvenv-$version"

eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

if [[ $(pyenv virtualenvs) == *"$venv"* ]]; then
    pyenv activate $venv
    python --version
    pip --version
else
    if [[ $(pyenv versions) != *"$version"* ]]; then
        pyenv install $version
    fi
    pyenv virtualenv $version $venv
    pyenv activate $venv
    python --version
    pip --version
    pip install numpy
    pip install pandas
    pip install scikit-learn
    pip install mock
    pip install pylint
    pip install autopep8
    pip install tensorflow
fi

python -m unittest discover -v -s graphsignal -p *_test.py

pyenv deactivate
