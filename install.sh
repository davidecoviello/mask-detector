#!/bin/bash

python3 -m venv venv
. venv/bin/activate
./venv/bin/python3 -m pip install --upgrade pip
./venv/bin/python3 -m pip install -r requirements.txt
