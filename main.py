import sys
import os
import logging
from pathlib import Path
from contextlib import contextmanager

from tts.audio_player import play_book, play_from_stdin
from tts.audio_processor import generate_audio_from_file
from utils.arg_parser import parse_arguments, validate_arguments
from config.settings import MODELS_DIR
from tts.voice_utils import get_available_voices
from utils.logger import setup_logger
from exceptions import (
    TTSError, VoiceNotFoundError, InputFileError, 
    AudioProcessingError, AudioPlaybackError, StdinProcessingError
)

logger = setup_logger()

@contextmanager
def resource_cleanup():
    """Context manager for cleanup of system resources"""
    try:
        yield
    finally:
        try:
            # Cleanup sound device
            import sounddevice as sd
            sd.stop()
        except Exception as e:
            logger.debug(f"Cleanup error: {e}")

def check_system_requirements():
    """Verify system requirements are met"""
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        if not devices:
            raise AudioPlaybackError("No audio output devices found")
    except Exception as e:
        raise AudioPlaybackError(f"Audio system initialization failed: {e}")

def list_available_voices():
    """List available voices and handle empty case"""
    voices = get_available_voices()
    if voices:
        logger.info("Available voices:")
        for voice in voices:
            logger.info(f"  {voice}")
        return True
    else:
        logger.warning(f"No voices found. Please download voices to: {MODELS_DIR}")
        return False

def main():
    """Main entry point for the script"""
    try:
        # Check system requirements
        check_system_requirements()
        
        # Parse arguments
        args = parse_arguments()
        
        # Handle voice listing
        if args.list_voices:
            success = list_available_voices()
            sys.exit(0 if success else 1)

        # Validate input requirements
        if not args.input_file and not args.stdin:
            logger.error("No input source specified")
            sys.exit("Error: either input_file or --stdin is required")

        # Validate arguments
        try:
            validate_arguments(args)
        except ValueError as e:
            logger.error(str(e))
            sys.exit(1)

        # Main processing
        with resource_cleanup():
            if args.stdin:
                logger.info("Processing input from stdin")
                try:
                    play_from_stdin(args.voice, args.speed)
                except Exception as e:
                    raise StdinProcessingError(f"Failed to process stdin: {e}")
                
            elif args.generate_audio:
                logger.info(f"Generating audio file: {args.generate_audio}")
                try:
                    generate_audio_from_file(
                        args.input_file, args.voice, args.speed, args.generate_audio
                    )
                except Exception as e:
                    raise AudioProcessingError(f"Failed to generate audio: {e}")
                
            else:
                logger.info(f"Playing audio from file: {args.input_file}")
                try:
                    play_book(args.input_file, args.voice, args.speed, args.save_audio)
                except Exception as e:
                    raise AudioPlaybackError(f"Failed to play audio: {e}")

    except KeyboardInterrupt:
        logger.info("\nProcess interrupted by user")
        sys.exit(1)
        
    except TTSError as e:
        logger.error(str(e))
        logger.debug("Detailed error:", exc_info=True)
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        logger.debug("Detailed error:", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
