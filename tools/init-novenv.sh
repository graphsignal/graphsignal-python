#!/bin/bash

set -e 

pip install -U poetry
poetry config virtualenvs.create false
poetry install
