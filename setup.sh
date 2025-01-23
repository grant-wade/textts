#!/bin/bash

# Create virtual environment
python3 -m venv venv

# Activate venv and install packages
source venv/bin/activate
pip install --upgrade pip
pip install onnxruntime phonemizer torch numpy

# Check for git-lfs installation
if ! command -v git-lfs &>/dev/null; then
    echo "git-lfs not found. Attempting to install..."
    git lfs install
    echo "git-lfs installed!"
fi

# Download kokomo voice model with LFS
git clone https://huggingface.co/hexgrad/Kokoro-82M

echo ""
echo "Setup complete! Activate the virtual environment with:"
echo "source venv/bin/activate"
