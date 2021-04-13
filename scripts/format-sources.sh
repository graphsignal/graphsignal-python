#!/bin/bash

source venv/bin/activate
autopep8 --verbose --in-place --aggressive --recursive graphsignal
deactivate