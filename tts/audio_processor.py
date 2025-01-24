import os
import time
import numpy as np
from tts.audio_generator import AudioGenerator
from utils.progress_tracker import FileReadingProgress
from config.settings import MODELS_DIR
from tts.voice_utils import get_available_voices
from tts.utilities import stream_sentences, save_audio_to_wav

def generate_audio_from_file(input_path, voice=None, speed=1.0, output_file="output.wav"):
    """Generate audio from a text file and save it to a WAV file"""
    sample_rate = int(22050 * speed)
    total_file_size = os.path.getsize(input_path)
    total_samples = 0
    sentences_processed = 0
    start_time = time.time()

    voices = get_available_voices()
    if not voices:
        print("Error: No voices found. Please download voices to:")
        print(MODELS_DIR)
        return

    selected_voice = voice if voice else voices[0]
    audio_gen = AudioGeneratorSync(selected_voice, speed=speed)
    progress = FileReadingProgress(input_path)
    sentence_stream = stream_sentences(input_path)
    audio_buffer = []

    try:
        print(f"Generating audio from {os.path.basename(input_path)}")
        print(f"Total file size: {total_file_size/1024:.1f} KB")
        print("=" * 50)

        # Skip to the saved progress position
        sentence_index = 0
        while sentence_index < progress.get_progress():
            next(sentence_stream, None)
            sentence_index += 1

        # Main audio generation loop
        for sentence in sentence_stream:
            # Process the sentence synchronously
            audio_gen.add_sentence(sentence)
            audio = audio_gen.get_audio()

            if audio is not None and len(audio) > 0:
                audio_buffer.append(audio)
                total_samples += len(audio)
                sentences_processed += 1

                # Calculate progress metrics
                elapsed = time.time() - start_time
                samples_sec = total_samples / elapsed if elapsed > 0 else 0
                bytes_total = total_samples * 2

                # Format progress output
                progress = (
                    f"\rProcessed: {sentences_processed} sentences "
                    f"| Audio: {total_samples//1000}k samples ({bytes_total//1024} KB) "
                    f"| Speed: {samples_sec//1000:.1f}k samples/s"
                )
                print(progress, end="", flush=True)

            # Update progress
            progress.update_progress(sentence_index)
            sentence_index += 1

    except Exception as e:
        print(f"Error generating audio: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    finally:
        print(f"\n\nProcessing complete!")
        print(f"Total sentences: {sentences_processed}")
        print(f"Final audio length: {total_samples/sample_rate:.1f} seconds")

        # Save the final audio
        if audio_buffer:
            save_audio_to_wav(output_file, audio_buffer, sample_rate)
