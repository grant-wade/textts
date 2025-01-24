import os
import time
import numpy as np
import sounddevice as sd
from pathlib import Path
from utils.progress_tracker import FileReadingProgress
from tts.audio_generator import AudioGenerator
from config.settings import MODELS_DIR
from tts.voice_utils import get_available_voices
from tts.utilities import stream_sentences, save_audio_to_wav

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
            sd.wait()  # Wait for playback to finish
            event.set()  # Signal completion
    except Exception as e:
        print(f"Error playing audio: {e}")

    if return_audio:
        return audio


def play_from_stdin(voice=None, speed=1.0):
    """Stream and play text from stdin using TTS"""
    voices = get_available_voices()
    if not voices:
        print("Error: No voices found. Please download voices to:")
        print(MODELS_DIR)
        return

    selected_voice = voice if voice else voices[0]
    audio_gen = AudioGenerator(selected_voice)
    sample_rate = int(22050 * speed)

    try:
        # Pre-fill the pipeline
        sentences = []
        sentence_stream = stream_from_stdin()
        
        # Pre-fill with initial sentences
        for _ in range(10):  # Pre-fill buffer
            sentence = next(sentence_stream, None)
            if sentence:
                sentences.append(sentence)
                audio_gen.add_sentence(sentence)

        # Main playback loop
        stream_exhausted = False
        while True:
            # Play available audio
            audio = audio_gen.get_audio()
            if audio is not None and len(audio) > 0:
                played_sentence = sentences.pop(0)
                print(f"\n{played_sentence}")
                play_audio(audio, audio_gen.audio_done_event, sample_rate=sample_rate)
                audio_gen.audio_done_event.wait(10)

            # Get next sentences until queue is full or stream exhausted
            while not stream_exhausted and audio_gen.sentence_queue.qsize() < 15:
                sentence = next(sentence_stream, None)
                if sentence:
                    sentences.append(sentence)
                    audio_gen.add_sentence(sentence)
                else:
                    stream_exhausted = True

            # Exit condition
            if stream_exhausted and not audio_gen.is_processing() and len(sentences) == 0:
                break

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nStopping playback...")
    except Exception as e:
        print(f"Error playing from stdin: {e}")
        import traceback
        traceback.print_exc()
    finally:
        audio_gen.stop()

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
        # Set up audio saving if requested
        output_file = None
        if save_audio:
            base_name = os.path.splitext(os.path.basename(input_path))[0]
            output_file = Path(f"{base_name}_audio") / f"{base_name}.wav"
            output_file.parent.mkdir(exist_ok=True, parents=True)

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

        # Display initial status
        if sentences:
            print(f"\nPre-filling audio queue with {len(sentences)} sentences...")

        # Start with the first sentence
        current_sentence = sentences[0] if sentences else None

        # Main playback loop
        stream_exhausted = False
        while True:
            # Play any available audio
            audio = audio_gen.get_audio()
            if audio is not None and len(audio) > 0:
                played_sentence = sentences.pop(0)
                # Display the sentence being played
                print(f"\n{played_sentence}")
                played_audio = play_audio(
                    audio,
                    audio_gen.audio_done_event,
                    sample_rate=sample_rate,
                    return_audio=True,
                )


                if played_audio is not None:
                    audio_buffer.append(played_audio)

                # Wait for current audio to finish with timeout
                audio_gen.audio_done_event.wait(10)

                # Update progress after audio completes
                progress.update_progress(sentence_index)
                sentence_index += 1

            # Get next sentences until queue is full or stream exhausted
            while not stream_exhausted and audio_gen.sentence_queue.qsize() < 15:
                current_sentence = next(sentence_stream, None)
                if current_sentence:
                    sentences.append(current_sentence)
                    audio_gen.add_sentence(current_sentence)
                else:
                    stream_exhausted = True

            # Exit condition: All streams processed and queues empty
            if (
                stream_exhausted
                and not audio_gen.is_processing()
                and len(sentences) == 0
            ):
                break

            # Prevent CPU spin
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
        if audio_buffer and output_file:
            save_audio_to_wav(output_file, audio_buffer, sample_rate)

        audio_gen.stop()
