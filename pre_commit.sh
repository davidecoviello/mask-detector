#!/bin/zsh

DEST_DIR=../../notebook/py

jupytext --to $DEST_DIR//py notebook/work/*.ipynb
