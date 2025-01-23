import argparse
import sys
import os
from tts.voice_utils import get_available_voices

def validate_arguments(args):
    """Validate the provided arguments"""
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found")
        sys.exit(1)

    if args.voice and args.voice not in get_available_voices():
        print(f"Error: Voice '{args.voice}' not found")
        sys.exit(1)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Split book into pages and optionally play them using TTS"
    )
    parser.add_argument("input_file", nargs="?", help="Path to the input text file")
    parser.add_argument("--voice", help="Voice to use for TTS (optional)")
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed multiplier (e.g. 1.5 for 50%% faster)",
    )
    parser.add_argument(
        "--list-voices",
        action="store_true",
        help="List all available voice models and exit",
    )
    parser.add_argument(
        "--save-audio",
        action="store_true",
        help="Save generated audio to WAV files in a directory named after the input file",
    )
    parser.add_argument(
        "--generate-audio",
        help="Generate audio from the input file and save it to the specified output file",
    )
    return parser.parse_args()
