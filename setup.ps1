# Create virtual environment
python -m venv venv

# Activate venv and install packages
.\venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install onnxruntime phonemizer torch numpy

# Check for git-lfs 
if (-not (Get-Command git-lfs -ErrorAction SilentlyContinue)) {
    Write-Host "git-lfs not found. Attempting to install..."
    git lfs install
    Write-Host "git-lfs installed!"
}

# Download kokomo voice model with LFS
git clone https://huggingface.co/hexgrad/Kokoro-82M

Write-Host ""
Write-Host "Setup complete! Activate the virtual environment with:"
Write-Host ".\venv\Scripts\Activate.ps1"
