import re
import wave
import numpy as np
from pathlib import Path

def stream_sentences(input_path):
    """Stream sentences from input file"""
    with open(input_path, "r", encoding="utf-8") as f:
        buffer = ""
        while True:
            chunk = f.read(4096)
            if not chunk:
                break
            chunk = chunk.replace("- ", "")
            buffer += chunk
            sentences = re.split(r"(?<=[.!?])\s+", buffer)
            buffer = sentences.pop(-1) if len(sentences) > 1 else buffer
            for sentence in sentences:
                if sentence.strip():
                    cleaned = re.sub(r"\n+", "", sentence.strip())
                    yield cleaned
        if buffer.strip():
            yield re.sub(r"\n+", " ", buffer.strip())

def save_audio_to_wav(output_file, audio_buffer, sample_rate):
    """Save audio buffer to a WAV file"""
    with wave.open(output_file, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        concatenated = np.concatenate(audio_buffer).astype(np.float32)
        wf.writeframes((concatenated * 32767).astype(np.int16).tobytes())
