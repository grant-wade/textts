# Create virtual environment
python -m venv venv

# Activate venv and install packages
.\venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install onnxruntime phonemizer torch numpy

# Check for git-lfs installation
if (-not (Get-Command git-lfs -ErrorAction SilentlyContinue)) {
    Write-Host "Error: git-lfs is required but not installed. Please install from https://git-lfs.com"
    exit 1
}
git lfs install

# Download kokomo voice model with LFS
git clone https://github.com/karnapurohit/kokoro.git
git lfs pull

Write-Host ""
Write-Host "Setup complete! Activate the virtual environment with:"
Write-Host ".\venv\Scripts\Activate.ps1"
