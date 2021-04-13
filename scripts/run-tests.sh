#!/bin/bash

source venv/bin/activate
python -m unittest discover -v -s graphsignal -p *_test.py
deactivate