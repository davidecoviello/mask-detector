#!/bin/zsh

DEST_DIR=../../notebook/work

jupytext --to $DEST_DIR//ipynb notebook/py/*.py
