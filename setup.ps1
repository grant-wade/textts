# Create virtual environment
python -m venv venv

# Activate venv and install packages
.\venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install onnxruntime phonemizer torch numpy

# Download kokomo voice model
git clone https://github.com/karnapurohit/kokoro.git

Write-Host ""
Write-Host "Setup complete! Activate the virtual environment with:"
Write-Host ".\venv\Scripts\Activate.ps1"
