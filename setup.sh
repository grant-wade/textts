#!/bin/bash

# Create virtual environment
python3 -m venv venv

# Activate venv and install packages
source venv/bin/activate
pip install --upgrade pip
pip install onnxruntime phonemizer torch numpy

# check for git lfs

# Download kokomo voice model
git clone https://huggingface.co/hexgrad/Kokoro-82M

echo ""
echo "Setup complete! Activate the virtual environment with:"
echo "source venv/bin/activate"
