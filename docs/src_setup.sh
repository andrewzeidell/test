#!/bin/bash

cd /workspaces/ML_Project/src
pip install --upgrade pip setuptools wheel\
	    && pip install -e ".[dev]"
