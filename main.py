import os
import re
import sys
import argparse
import time
import numpy as np
import wave
import os
import sounddevice as sd
from pathlib import Path
from utils.progress_tracker import FileReadingProgress
from tts.audio_generator import AudioGenerator

VOICE_NAME = "af"

# Configuration
KOKORO_PATH = Path(".") / "kokoro"
MODELS_DIR = KOKORO_PATH / "voices"



def get_available_voices():
    """Get list of available Piper voices"""
    voices = []
    if MODELS_DIR.exists():
        for model in MODELS_DIR.glob("*.pt"):
            voices.append(model.stem)
    return voices


def play_audio(audio, event, sample_rate=22050, return_audio=False):
    """Play audio array using sounddevice"""
    try:
        # Ensure audio is in the correct format (mono, float32)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)  # Convert to mono if stereo
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Normalize audio to prevent clipping
        audio_max = np.max(np.abs(audio))
        if audio_max > 0:
            audio /= audio_max

        # Play audio using blocking playback with safety checks
        if audio.size > 0:
            sd.play(audio, samplerate=sample_rate, blocking=False)
            event.clear()  # Clear event at start of playback
            sd.wait()      # Wait for playback to finish
            event.set()    # Signal completion
    except Exception as e:
        print(f"Error playing audio: {e}")
    
    if return_audio:
        return audio





def stream_sentences(input_path):
    """Stream sentences from input file"""
    with open(input_path, "r", encoding="utf-8") as f:
        buffer = ""
        while True:
            chunk = f.read(4096)  # Read in chunks
            if not chunk:
                break
            # Remove "- " strings from the text
            chunk = chunk.replace("- ", "")
            buffer += chunk
            # Split on sentence boundaries
            sentences = re.split(r"(?<=[.!?])\s+", buffer)
            # Keep last partial sentence in buffer
            buffer = sentences.pop(-1) if len(sentences) > 1 else buffer
            for sentence in sentences:
                if sentence.strip():  # Skip empty sentences
                    # Remove extra newlines while preserving sentence structure
                    cleaned = re.sub(r"\n+", "", sentence.strip())
                    yield cleaned
        # Yield any remaining text
        if buffer.strip():
            yield re.sub(r"\n+", " ", buffer.strip())


def play_book(input_path, voice=None, speed=1.0, save_audio=False):
    # Set up audio saving if requested
    audio_buffer = []
    output_dir = None
    file_counter = 1
    sample_rate = int(22050 * speed)
    samples_per_minute = sample_rate * 60
    
    if save_audio:
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_dir = Path(f"{base_name}_audio")
        output_dir.mkdir(exist_ok=True)
    """Stream and play a book using TTS"""
    voices = get_available_voices()
    if not voices:
        print("Error: No voices found. Please download voices to:")
        print(MODELS_DIR)
        return

    selected_voice = voice if voice else voices[0]
    audio_gen = AudioGenerator(selected_voice)
    progress = FileReadingProgress(input_path)
    sentence_stream = stream_sentences(input_path)

    try:
        # Skip to the saved progress position
        sentence_index = 0
        while sentence_index < progress.get_progress():
            next(sentence_stream, None)
            sentence_index += 1
        
        # Pre-fill the pipeline with more sentences
        prefill_count = 10  # Increased pre-fill buffer
        sentences = []
        for _ in range(prefill_count):
            sentence = next(sentence_stream, None)
            if sentence:
                sentences.append(sentence)
                audio_gen.add_sentence(sentence
        
        # Display initial status
        if sentences:
            print(f"\nPre-filling audio queue with {len(sentences)} sentences...")
    
        # Start with the first sentence
        current_sentence = sentences[0] if sentences else None

        # Main playback loop
        while len(sentences) > 0 or not audio_gen.audio_queue.empty():
            # Play any available audio
            audio = audio_gen.get_audio()
            if audio is not None and len(audio) > 0:
                played_sentence = sentences.pop(0)
                print(f"{played_sentence}\n")
                played_audio = play_audio(
                    audio, 
                    audio_gen.audio_done_event, 
                    sample_rate=sample_rate,
                    return_audio=save_audio
                )
                
                if save_audio and played_audio is not None:
                    audio_buffer.append(played_audio)
                    # Save when we have at least 1 minute of audio
                    if sum(len(chunk) for chunk in audio_buffer) >= samples_per_minute:
                        output_file = output_dir / f"{file_counter:04d}.wav"
                        with wave.open(str(output_file), "wb") as wf:
                            wf.setnchannels(1)
                            wf.setsampwidth(2)
                            wf.setframerate(sample_rate)
                            concatenated = np.concatenate(audio_buffer).astype(np.float32)
                            wf.writeframes((concatenated * 32767).astype(np.int16).tobytes())
                        audio_buffer = []
                        file_counter += 1
                
                # Wait for current audio to finish with timeout
                audio_gen.audio_done_event.wait(10)  # 10 second timeout
                
                # Update progress after audio completes
                progress.update_progress(sentence_index)
                sentence_index += 1

            # Get next sentence from stream if available
            current_sentence = next(sentence_stream, None)
            if current_sentence:
                sentences.append(current_sentence)
                audio_gen.add_sentence(current_sentence)
            
            # Don't spin too fast if queue is empty
            if len(sentences) == 0 and not audio_gen.stop_event.is_set():
                time.sleep(0.1)

        # Final wait for last audio to finish
        audio_gen.audio_done_event.wait()

    except KeyboardInterrupt:
        print("\nStopping playback...")
        progress.save_progress()
        print(f"Progress saved at sentence {progress.get_progress()}")
        exit(1)
    except Exception as e:
        print(f"Error playing page: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    finally:
        # Save any remaining audio
        if save_audio and audio_buffer:
            output_file = output_dir / f"{file_counter:04d}.wav"
            with wave.open(str(output_file), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                concatenated = np.concatenate(audio_buffer).astype(np.float32)
                wf.writeframes((concatenated * 32767).astype(np.int16).tobytes())
        
        audio_gen.stop()


def validate_arguments(args):
    """Validate the provided arguments"""
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found")
        sys.exit(1)

    if args.voice and args.voice not in get_available_voices():
        print(f"Error: Voice '{args.voice}' not found")
        sys.exit(1)


def main():
    """Main entry point for the script"""
    parser = argparse.ArgumentParser(
        description="Split book into pages and optionally play them using Piper TTS"
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

    args = parser.parse_args()
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
        parser.error("the following arguments are required: input_file")

    if not args.list_voices:
        validate_arguments(args)
        play_book(
            args.input_file, 
            args.voice, 
            args.speed,
            args.save_audio
        )


if __name__ == "__main__":
    main()
