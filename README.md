# Audiobook Generator with [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)

## Introduction

A text-to-speech audiobook generator that uses Kokoro-82M for high-quality voice synthesis. Features include:

- Progress tracking and resuming for playback
- Cross-platform support (tested on linux)
- Adjustable playback speed
- Save generated audio to WAV files

## Installation

### Prerequisites

- Python 3.8+
- git
- git-lfs
- espeak (for phonemizer)

### Setup Steps

#### Windows

```powershell
.\setup.ps1
```

#### Linux/macOS

```bash
bash setup.sh
```

After setup, activate the virtual environment:

- Windows: `.\venv\Scripts\Activate.ps1`
- Linux/macOS: `source venv/bin/activate`

## Usage

Basic command:

```bash
python main.py [input_file.txt] [options]
```

Options:

- `--voice [VOICE_NAME]`: Select TTS voice (default: af)
- `--speed [MULTIPLIER]`: Adjust playback speed (default: 1.0)
- `--list-voices`: Show available voices (used on it's own)
- `--save-audio`: Save generated audio to WAV files
- `--generate-audio [OUTPUT_FILE]`: Generate audio from the input file and save to specified output file

Examples:

Play book with options:
```bash
python main.py book.txt --voice af --speed 1.2 --save-audio
```

Generate audio file:
```bash
python main.py book.txt --voice af --speed 1.0 --generate-audio output.wav
```

List available voices:
```bash
python main.py --list-voices
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
