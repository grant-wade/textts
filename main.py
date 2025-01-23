import sys
from tts.audio_player import play_book
from tts.audio_processor import generate_audio_from_file
from utils.arg_parser import parse_arguments, validate_arguments
from config.settings import MODELS_DIR
from tts.voice_utils import get_available_voices

def main():
    """Main entry point for the script"""
    args = parse_arguments()
    
    if args.list_voices:
        voices = get_available_voices()
        if voices:
            print("Available voices:")
            for voice in voices:
                print(f"  {voice}")
        else:
            print(f"No voices found. Please download voices to: {MODELS_DIR}")
        sys.exit(0)

    if not args.list_voices and not args.input_file:
        sys.exit("Error: the following arguments are required: input_file")

    if not args.list_voices:
        validate_arguments(args)

        if args.generate_audio:
            generate_audio_from_file(
                args.input_file, args.voice, args.speed, args.generate_audio
            )
        else:
            play_book(args.input_file, args.voice, args.speed, args.save_audio)

if __name__ == "__main__":
    main()
