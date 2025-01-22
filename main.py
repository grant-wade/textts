import os
import re
import sys
import subprocess
import argparse
from pathlib import Path

# Configuration
PIPER_PATH = Path.home() / "LLM" / "piper_amd64" / "piper"
MODELS_DIR = Path.home() / "Downloads"


def get_available_voices():
    """Get list of available Piper voices"""
    voices = []
    if MODELS_DIR.exists():
        for model in MODELS_DIR.glob("*.onnx"):
            voices.append(model.stem)
    return voices


def check_piper_installed():
    """Check if Piper is installed and available"""
    try:
        subprocess.run(
            ["piper", "--version"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def play_page(page_path, voice=None):
    """Play a page using Piper TTS piped to aplay"""
    if not check_piper_installed():
        print("Error: Piper TTS is not installed or not in PATH")
        print("Please install Piper first: https://github.com/rhasspy/piper")
        return

    voices = get_available_voices()
    if not voices:
        print("Error: No Piper voices found. Please download voices to:")
        print(MODELS_DIR)
        return

    # Use first voice by default if none specified
    selected_voice = voice if voice else voices[0]

    try:
        # Clear terminal and display page text
        os.system("clear")
        
        # Extract page number from filename (format: page_XXX.txt)
        page_num = int(Path(page_path).stem.split('_')[-1])
        
        with open(page_path, "r", encoding="utf-8") as f:
            page_text = f.read()
            # Clean up text: remove single newlines and normalize spaces
            cleaned_text = re.sub(
                r"(?<!\n)\n(?!\n)", " ", page_text
            )  # Single newlines to spaces
            cleaned_text = re.sub(
                r"[ \t]+", " ", cleaned_text
            )  # Multiple spaces/tabs to single space
            
            # Display page number and text
            print(f"\n=== Page {page_num} ===\n")
            print(cleaned_text)
            print(f"\n=== End of Page {page_num} ===\n")

        piper_cmd = [
            "piper",
            "--model",
            str(MODELS_DIR / f"{selected_voice}.onnx"),
            "--output-raw",
        ]
        aplay_cmd = ["aplay", "-r", "22050", "-f", "S16_LE", "-t", "raw", "-c", "1"]

        # Pipe Piper output to aplay with proper resource management
        with subprocess.Popen(
            piper_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        ) as piper_process, subprocess.Popen(
            aplay_cmd,
            stdin=piper_process.stdout,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ) as aplay_process:
            # Send page text to Piper
            try:
                piper_process.stdin.write(cleaned_text.encode())
                piper_process.stdin.close()

                # Wait for playback to finish
                aplay_process.wait()
            except Exception as e:
                # Ensure processes are terminated if an error occurs
                piper_process.terminate()
                aplay_process.terminate()
                raise
    except Exception as e:
        print(f"Error playing page: {e}")


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


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Split book into pages and optionally play them using Piper TTS"
    )
    parser.add_argument("input_file", help="Path to the input text file")
    parser.add_argument(
        "page", nargs="?", type=int, help="Page number to play (optional)"
    )
    parser.add_argument("--voice", help="Voice to use for TTS (optional)")
    parser.add_argument(
        "--continue",
        action="store_true",
        help="Continue playing subsequent pages after the specified page",
    )

    # Add available voices to help text
    voices = get_available_voices()
    if voices:
        parser.epilog = "Available voices:\n  " + "\n  ".join(voices)
    else:
        parser.epilog = "No voices found. Please download voices to: " + str(MODELS_DIR)

    return parser.parse_args()


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
    args = parse_arguments()
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
                print(f"Playing page {current_page}...")
                play_page(page_path, args.voice)
                current_page += 1
        else:
            # Play just the specified page
            page_path = os.path.join(output_dir, f"page_{args.page:03d}.txt")
            if not os.path.exists(page_path):
                print(f"Error: Page {args.page} not found")
                sys.exit(1)
            play_page(page_path, args.voice)


if __name__ == "__main__":
    main()
