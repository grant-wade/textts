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
    audio_gen = AudioGenerator(selected_voice)
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

        # Pre-fill the pipeline with more sentences
        prefill_count = 10  # Increased pre-fill buffer
        sentences = []
        for _ in range(prefill_count):
            sentence = next(sentence_stream, None)
            if sentence:
                sentences.append(sentence)
                audio_gen.add_sentence(sentence)

        # Start with the first sentence
        current_sentence = sentences[0] if sentences else None

        # Main audio generation loop
        while audio_gen.sentence_queue.qsize() > 0 or audio_gen.audio_queue.qsize() > 0:
            if len(sentences) > 0:
                sentences.pop(0)

            audio = audio_gen.get_audio()
            if audio is not None and len(audio) > 0:
                audio_buffer.append(audio)
                total_samples += len(audio)
                sentences_processed += 1

                # Calculate progress metrics
                elapsed = time.time() - start_time
                samples_sec = total_samples / elapsed if elapsed > 0 else 0
                bytes_total = total_samples * 2
                progress_percent = (
                    sentences_processed
                    / (sentences_processed + audio_gen.sentence_queue.qsize())
                ) * 100

                # Format progress output
                progress = (
                    f"\rProcessed: {sentences_processed} sentences "
                    f"| Audio: {total_samples//1000}k samples ({bytes_total//1024} KB) "
                    f"| Speed: {samples_sec//1000:.1f}k samples/s "
                    f"| Progress: {progress_percent:.1f}%"
                )
                print(progress, end="", flush=True)

            # Get next sentence from stream if available
            current_sentence = next(sentence_stream, None)
            if current_sentence:
                sentences.append(current_sentence)
                audio_gen.add_sentence(current_sentence)

            # Don't spin too fast if queue is empty
            if len(sentences) == 0 and not audio_gen.stop_event.is_set():
                time.sleep(0.1)

        # Final wait for last audio to finish
        while audio_gen.audio_queue.qsize() > 0:
            time.sleep(0.1)

        # Stop the audio generator
        print("\n\nStopping audio generator")
        audio_gen.stop()
        print("Audio generator stopped")

    except Exception as e:
        print(f"Error generating audio: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    finally:
        print(f"\n\nProcessing complete!")
        print(f"Total sentences: {sentences_processed}")
        print(f"Final audio length: {total_samples/sample_rate:.1f} seconds")

        # Save any remaining audio
        if audio_buffer:
            save_audio_to_wav(output_file, audio_buffer, sample_rate)

        # Request stop and wait with timeout
        audio_gen.stop()

