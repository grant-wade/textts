# Audiobook Generator with Piper TTS

## Introduction
A text-to-speech audiobook generator that uses Piper TTS for high-quality voice synthesis. Features include:
- Book splitting into manageable pages
- Progress tracking and resuming
- Cross-platform support
- Adjustable playback speed
- Audio context preservation between pages
- Save generated audio to WAV files

## Installation

### Prerequisites
- Python 3.8+
- git
- git-lfs

### Setup Steps

#### Windows
```powershell
./setup.ps1
```

#### Linux/macOS
```bash
chmod +x setup.sh
./setup.sh
```

After setup, activate the virtual environment:
- Windows: `.\venv\Scripts\Activate.ps1`
- Linux/macOS: `source venv/bin/activate`

Download voice models using git-lfs:
```bash
git clone https://huggingface.co/hexgrad/Kokoro-82M
```

## Usage

Basic command:
```bash
python main.py [input_file.txt] [options]
```

Options:
- `--voice [VOICE_NAME]`: Select TTS voice (default: af)
- `--speed [MULTIPLIER]`: Adjust playback speed (default: 1.0)
- `--context`: Show context from previous/next pages
- `--list-voices`: Show available voices
- `--save-audio`: Save generated audio to WAV files

Example with options:
```bash
python main.py book.txt --voice kokoro --speed 1.2 --context --save-audio
```

## Technology Stack
- **Core TTS Engine**: Piper TTS
- **Text Processing**: Custom sentence segmentation and context preservation
- **Audio Playback**: sounddevice library with numpy-based processing
- **Multithreading**: Queued audio generation/playback
- **Progress Tracking**: File-based position memory
- **Dependencies**:
  - ONNX Runtime for inference
  - Phonemizer for text normalization
  - Torch for backend processing
  - NumPy for audio processing
- **Voice Models**: ONNX format voice models (Kokoro-82M by default)
