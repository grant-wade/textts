#!/bin/bash

# Create virtual environment
python3 -m venv venv

# Activate venv and install packages
source venv/bin/activate
pip install --upgrade pip
pip install onnxruntime phonemizer torch numpy

# Check for git-lfs installation
if ! command -v git-lfs &> /dev/null; then
    echo "Error: git-lfs is required but not installed. Please install from https://git-lfs.com"
    exit 1
fi
git lfs install

# Download kokomo voice model with LFS
git clone https://huggingface.co/hexgrad/Kokoro-82M
git lfs pull

echo ""
echo "Setup complete! Activate the virtual environment with:"
echo "source venv/bin/activate"
