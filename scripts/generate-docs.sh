#!/bin/bash

source venv/bin/activate
pydoc-markdown -I graphsignal -m . > /tmp/api-reference.md
pydoc-markdown -I graphsignal -m sessions >> /tmp/api-reference.md
deactivate
