import sys
import os
import logging
from pathlib import Path
from contextlib import contextmanager

from tts.engine import TTSEngine, TTSConfig
from tts.audio_player import play_audio
from utils.arg_parser import parse_arguments, validate_arguments
from config.settings import MODELS_DIR
from tts.voice_utils import get_available_voices
from utils.logger import setup_logger
from tts.exceptions import (
    TTSException, ConfigurationError, EngineInitializationError,
    SpeechGenerationError, AudioPlaybackError
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

        # Create TTS engine
        config = TTSConfig(
            voice_name=args.voice,
            speed=args.speed
        )
        engine = TTSEngine(config)

        # Main processing
        with resource_cleanup():
            if args.stdin:
                logger.info("Processing input from stdin")
                try:
                    for line in sys.stdin:
                        engine.speak(line.strip(), mode="live")  # Use live mode for stdin
                except Exception as e:
                    raise SpeechGenerationError(f"Failed to process stdin: {e}")
                
            elif args.generate_audio:
                logger.info(f"Generating audio file: {args.generate_audio}")
                try:
                    engine.save_to_file(args.input_file, args.generate_audio)
                except Exception as e:
                    raise SpeechGenerationError(f"Failed to generate audio: {e}")
                
            else:
                logger.info(f"Playing audio from file: {args.input_file}")
                try:
                    with open(args.input_file) as f:
                        for line in f:
                            engine.speak(line.strip(), mode="live")  # Use live mode for file reading
                except Exception as e:
                    raise SpeechGenerationError(f"Failed to play audio: {e}")

    except KeyboardInterrupt:
        logger.info("\nProcess interrupted by user")
        sys.exit(1)
        
    except TTSException as e:
        logger.error(str(e))
        logger.debug("Detailed error:", exc_info=True)
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        logger.debug("Detailed error:", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
