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

        # Play audio
        sd.play(audio, samplerate=sample_rate)
        sd.wait()  # Wait until audio is finished playing
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
        paragraphs = text.split('\n\n')
        
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


def play_page(page_path, voice=None, show_context=False, next_page_path=None):
    """Play a page using Piper TTS piped to aplay"""
    voices = get_available_voices()
    if not voices:
        print("Error: No voices found. Please download voices to:")
        print(MODELS_DIR)
        return

    # Use first voice by default if none specified
    selected_voice = voice if voice else voices[0]
    audio_gen = AudioGenerator(selected_voice)
    next_audio_gen = AudioGenerator(selected_voice) if next_page_path else None

    try:
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
            cleaned_text = re.sub(r"(?<!\n)\n(?!\n)", " ", page_text)  # Convert single newlines to spaces
            cleaned_text = re.sub(r"[ \t]+", " ", cleaned_text)  # Normalize spaces
            cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)  # Normalize multiple newlines
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

        # Start audio generation in background
        audio_gen.start_generation(cleaned_text)
        print("Generating audio... (press Ctrl+C to stop)")

        # If we have a next page, start preloading its audio
        if next_page_path:
            with open(next_page_path, "r", encoding="utf-8") as f:
                next_text = f.read()
                next_audio_gen.start_generation(next_text)

        # Play audio chunks as they become available
        while True:
            audio = audio_gen.get_next_audio()
            if audio is not None:
                play_audio(audio)

                # If we're near the end of current audio and have next page ready
                if (
                    next_audio_gen
                    and audio_gen.audio_queue.qsize() < 2
                    and not next_audio_gen.audio_queue.empty()
                ):
                    # Switch to next page's audio
                    audio_gen.stop()
                    audio_gen = next_audio_gen
                    next_audio_gen = None
            elif not audio_gen.worker_thread.is_alive():
                break
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nStopping playback...")
    except Exception as e:
        print(f"Error playing page: {e}")
    finally:
        audio_gen.stop()


def split_book_to_pages(input_path):
    # Create output directory based on input filename
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_dir = f"{base_name}_pages"
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Open the input file for reading
    with open(input_path, "r", encoding="utf-8") as infile:
        lines = infile.readlines()

    current_page = None
    current_file = None
    page_pattern = re.compile(r"^\d+\s*$")

    for line in lines:
        if page_pattern.match(line):
            # Close the current file if it exists
            if current_file is not None:
                current_file.close()
            # Increment page number
            current_page = 0 if current_page is None else current_page + 1
            # Create new filename with leading zeros to maintain order
            filename = os.path.join(output_dir, f"page_{current_page:03d}.txt")
            current_file = open(filename, "w", encoding="utf-8")
        else:
            # If we haven't found the first page marker yet, create page_000
            if current_page is None:
                current_page = 0
                filename = os.path.join(output_dir, f"page_{current_page:03d}.txt")
                current_file = open(filename, "w", encoding="utf-8")
            if current_file is not None:
                current_file.write(line)

    # Close the last file if it's open
    if current_file is not None:
        current_file.close()


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
        # Split the book into pages
    split_book_to_pages(args.input_file)

    # If a page number was provided, play that page
    if args.page is not None:
        base_name = os.path.splitext(os.path.basename(args.input_file))[0]
        output_dir = f"{base_name}_pages"

        # Play pages sequentially if --continue is set
        if getattr(args, "continue", False):
            current_page = args.page
            while True:
                page_path = os.path.join(output_dir, f"page_{current_page:03d}.txt")
                if not os.path.exists(page_path):
                    break
                # Get next page path if it exists
                next_page_path = os.path.join(
                    output_dir, f"page_{current_page+1:03d}.txt"
                )
                next_page_path = (
                    next_page_path if os.path.exists(next_page_path) else None
                )

                print(f"Playing page {current_page}...")
                play_page(page_path, args.voice, args.context, next_page_path)
                current_page += 1
        else:
            # Play just the specified page
            page_path = os.path.join(output_dir, f"page_{args.page:03d}.txt")
            if not os.path.exists(page_path):
                print(f"Error: Page {args.page} not found")
                sys.exit(1)
            play_page(page_path, args.voice, args.context)


if __name__ == "__main__":
    main()
