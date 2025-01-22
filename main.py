import os
import re
import sys
import subprocess
from pathlib import Path

# Configuration
PIPER_PATH = Path.home() / ".local" / "share" / "piper"
MODELS_DIR = PIPER_PATH / "models"


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
        subprocess.run(["piper", "--version"], check=True, 
                      stdout=subprocess.PIPE, stderr=subprocess.PIPE)
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
        piper_cmd = ["piper", "--model", str(MODELS_DIR / f"{selected_voice}.onnx"), "--output-raw"]
        aplay_cmd = ["aplay", "-r", "22050", "-f", "S16_LE", "-t", "raw", "-c", "1"]
        
        # Pipe Piper output to aplay with proper resource management
        with subprocess.Popen(piper_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE) as piper_process, \
             subprocess.Popen(aplay_cmd, stdin=piper_process.stdout) as aplay_process:
            
            # Send page text to Piper
            try:
                with open(page_path, "r", encoding="utf-8") as f:
                    piper_process.stdin.write(f.read().encode())
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


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <input_file> [play_page_number] [--voice VOICE_NAME]")
        print("\nAvailable voices:")
        voices = get_available_voices()
        if voices:
            for voice in voices:
                print(f"  - {voice}")
        else:
            print("  No voices found. Please download voices to:", MODELS_DIR)
        sys.exit(1)
    
    input_file = sys.argv[1]
    split_book_to_pages(input_file)
    
    # Parse optional arguments
    voice = None
    page_num = None
    
    if len(sys.argv) > 2:
        try:
            # Check for --voice argument
            if "--voice" in sys.argv:
                voice_index = sys.argv.index("--voice")
                if len(sys.argv) > voice_index + 1:
                    voice = sys.argv[voice_index + 1]
                    if voice not in get_available_voices():
                        print(f"Error: Voice '{voice}' not found")
                        sys.exit(1)
            
            # Get page number if provided
            if len(sys.argv) > 2 and sys.argv[2].isdigit():
                page_num = int(sys.argv[2])
                
            if page_num is not None:
                base_name = os.path.splitext(os.path.basename(input_file))[0]
                output_dir = f"{base_name}_pages"
                page_path = os.path.join(output_dir, f"page_{page_num:03d}.txt")
                
                if not os.path.exists(page_path):
                    print(f"Error: Page {page_num} not found")
                    sys.exit(1)
                    
                play_page(page_path, voice)
                
        except ValueError:
            print("Page number must be an integer")
