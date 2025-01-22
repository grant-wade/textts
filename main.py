import os
import re
import sys
import argparse
import numpy as np
from pathlib import Path
from kokoro.kokoro import phonemize, tokenize
from onnxruntime import InferenceSession
import torch
import sounddevice as sd
import threading
import queue
import time

device = "cuda" if torch.cuda.is_available() else "cpu"


VOICE_NAME = "af"
VOICEPACK = torch.load(f"kokoro/voices/{VOICE_NAME}.pt", weights_only=True).to(device)

# Configuration
KOKORO_PATH = Path.home() / "LLM" / "reader" / "kokoro"
MODELS_DIR = KOKORO_PATH / "voices"


def get_available_voices():
    """Get list of available Piper voices"""
    voices = []
    if MODELS_DIR.exists():
        for model in MODELS_DIR.glob("*.pt"):
            voices.append(model.stem)
    return voices


def play_audio(audio, sample_rate=22050):
    """Play audio array using sounddevice"""
    try:
        # Ensure audio is in the correct format (mono, float32)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)  # Convert to mono if stereo
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Normalize audio to prevent clipping
        audio /= np.max(np.abs(audio))

        # Play audio non-blocking
        sd.play(audio, samplerate=sample_rate, blocking=False)
    except Exception as e:
        print(f"Error playing audio: {e}")


def get_page_context(page_path, num_sentences=2):
    """Get the first or last few sentences from a page"""
    if not os.path.exists(page_path):
        return ""

    with open(page_path, "r", encoding="utf-8") as f:
        text = f.read()
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return " ".join(
            sentences[:num_sentences]
            if "prev" in str(page_path)
            else sentences[-num_sentences:]
        )


class AudioGenerator:
    def __init__(self, voice_name):
        self.voice_name = voice_name
        self.audio_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.worker_thread = None
        self.sess = InferenceSession("kokoro/kokoro-v0_19.onnx")

    def _generate_audio_batch(self, text):
        """Generate audio for a text batch in a thread"""
        # Split text into paragraphs first
        paragraphs = text.split("\n\n")

        for paragraph in paragraphs:
            # Split paragraph into sentences
            sentences = re.split(r"(?<=[.!?])\s+", paragraph)

            for sentence in sentences:
                # Skip empty or whitespace-only sentences
                if not sentence or sentence.isspace():
                    continue

                try:
                    # Process each sentence individually
                    phonemes = phonemize(sentence, self.voice_name[0])
                    tokens = tokenize(phonemes)
                    self._process_token_batch(tokens)
                except Exception as e:
                    print(f"Error processing sentence: {e}")
                    print(f"Sentence: {sentence}")
                    continue

    def _process_token_batch(self, tokens):
        """Process a batch of tokens and add to audio queue"""
        try:
            if not tokens:  # Skip empty token batches
                return

            ref_s = torch.load(
                f"kokoro/voices/{self.voice_name}.pt", weights_only=True
            )[len(tokens)].numpy()
            tokens = [[0, *tokens, 0]]
            audio = self.sess.run(
                None, dict(tokens=tokens, style=ref_s, speed=np.ones(1, np.float32))
            )[0]
            self.audio_queue.put(audio)
        except Exception as e:
            print(f"Error processing token batch: {e}")

    def start_generation(self, text):
        """Start audio generation in a background thread"""
        self.stop_event.clear()
        self.worker_thread = threading.Thread(
            target=self._generate_audio_batch, args=(text,)
        )
        self.worker_thread.start()

    def get_next_audio(self):
        """Get the next audio chunk from the queue"""
        try:
            return self.audio_queue.get(timeout=0.1)
        except queue.Empty:
            return None

    def stop(self):
        """Stop generation and clean up"""
        self.stop_event.set()
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join()
        while not self.audio_queue.empty():
            self.audio_queue.get()


def display_page(page_path, show_context=False):
    """Display page text with optional context"""
    # Extract page number from filename (format: page_XXX.txt)
    page_num = int(Path(page_path).stem.split("_")[-1])
    pages_dir = Path(page_path).parent

    with open(page_path, "r", encoding="utf-8") as f:
        if show_context:
            # Get previous and next page paths
            prev_page = pages_dir / f"page_{page_num-1:03d}.txt"
            next_page = pages_dir / f"page_{page_num+1:03d}.txt"

            # Get context from adjacent pages
            prev_context = get_page_context(prev_page)
            next_context = get_page_context(next_page, num_sentences=2)

        page_text = f.read()
        # Clean up text - preserve paragraph breaks but normalize other whitespace
        cleaned_text = re.sub(
            r"(?<!\n)\n(?!\n)", " ", page_text
        )  # Convert single newlines to spaces
        cleaned_text = re.sub(r"[ \t]+", " ", cleaned_text)  # Normalize spaces
        cleaned_text = re.sub(
            r"\n{3,}", "\n\n", cleaned_text
        )  # Normalize multiple newlines
        cleaned_text = cleaned_text.strip()

        # Display page text with optional context
        if show_context:
            if prev_context:
                print(f"\n[Previous page ending...]\n{prev_context}\n")

        print(f"\n=== Page {page_num} ===\n")
        print(f"{cleaned_text}")
        print(f"\n=== End of Page {page_num} ===\n")

        if show_context and next_context:
            print(f"\n[Next page starting...]\n{next_context}\n")

    return cleaned_text


def play_book(input_path, voice=None, show_context=False):
    """Stream and play a book using TTS"""
    voices = get_available_voices()
    if not voices:
        print("Error: No voices found. Please download voices to:")
        print(MODELS_DIR)
        return

    # Use first voice by default if none specified
    selected_voice = voice if voice else voices[0]
    audio_gen = AudioGenerator(selected_voice)

    # Buffer for upcoming sentences and their audio
    sentence_buffer = queue.Queue(maxsize=10)  # Keep 10 sentences ahead
    audio_buffer = queue.Queue(maxsize=10)  # Keep 10 audio chunks ahead
    stop_event = threading.Event()

    def buffer_sentences():
        """Fill the sentence buffer and pre-generate audio"""
        # Start generating audio for multiple sentences in parallel
        for sentence in stream_sentences(input_path):
            if stop_event.is_set():
                break
            sentence_buffer.put(sentence)
            audio_gen.start_generation(sentence)

        sentence_buffer.put(None)  # Signal end of stream

    def audio_worker():
        """Worker to continuously fetch and buffer audio"""
        while not stop_event.is_set():
            try:
                audio = audio_gen.get_next_audio()
                if audio is not None:
                    audio_buffer.put(audio)
                elif not audio_gen.worker_thread.is_alive():
                    break
            except queue.Empty:
                time.sleep(0.01)
        audio_buffer.put(None)  # Signal end of audio

    # Start buffering sentences and audio in background
    buffer_thread = threading.Thread(target=buffer_sentences)
    audio_worker_thread = threading.Thread(target=audio_worker)
    buffer_thread.start()
    audio_worker_thread.start()

    try:
        print("Starting playback... (press Ctrl+C to stop)")
        while True:
            sentence = sentence_buffer.get()
            if sentence is None:  # End of stream
                break

            # Display current sentence
            print(f"\n{sentence}\n")

            # Play pre-generated audio
            audio = audio_buffer.get()
            if audio is not None:
                play_audio(audio)

    except KeyboardInterrupt:
        print("\nStopping playback...")
    except Exception as e:
        print(f"Error playing page: {e}")
    finally:
        audio_gen.stop()


def stream_sentences(input_path):
    """Stream sentences from input file"""
    with open(input_path, "r", encoding="utf-8") as f:
        buffer = ""
        while True:
            chunk = f.read(4096)  # Read in chunks
            if not chunk:
                break
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


def validate_arguments(args):
    """Validate the provided arguments"""
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found")
        sys.exit(1)

    if args.voice and args.voice not in get_available_voices():
        print(f"Error: Voice '{args.voice}' not found")
        sys.exit(1)

    if args.page is not None and args.page < 0:
        print("Error: Page number must be positive")
        sys.exit(1)


def main():
    """Main entry point for the script"""
    parser = argparse.ArgumentParser(
        description="Split book into pages and optionally play them using Piper TTS"
    )
    parser.add_argument("input_file", nargs="?", help="Path to the input text file")
    parser.add_argument(
        "page", nargs="?", type=int, help="Page number to play (optional)"
    )
    parser.add_argument("--voice", help="Voice to use for TTS (optional)")
    parser.add_argument(
        "--continue",
        action="store_true",
        help="Continue playing subsequent pages after the specified page",
    )
    parser.add_argument(
        "--context",
        action="store_true",
        help="Show context from adjacent pages",
    )
    parser.add_argument(
        "--list-voices",
        action="store_true",
        help="List all available voice models and exit",
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
        play_book(args.input_file, args.voice, args.context)


if __name__ == "__main__":
    main()
