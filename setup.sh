#!/bin/bash

# Create virtual environment
python3 -m venv venv

# Activate venv and install packages
source venv/bin/activate
pip install --upgrade pip
pip install onnxruntime phonemizer torch numpy

# Download kokomo voice model
git clone https://github.com/karnapurohit/kokoro.git

echo ""
echo "Setup complete! Activate the virtual environment with:"
echo "source venv/bin/activate"
