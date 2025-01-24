import numpy as np
import sounddevice as sd

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
