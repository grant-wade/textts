from typing import Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path

@dataclass
class TTSConfig:
    voice_name: Optional[str] = None
    speed: float = 1.0
    volume: float = 1.0
    sample_rate: int = 22050
    models_dir: Optional[Path] = None
    cache_dir: Optional[Path] = None

class TTSEngine:
    def __init__(self, config: Optional[TTSConfig] = None):
        self.config = config or TTSConfig()
        self._initialize_engine()
    
    def _initialize_engine(self):
        """Initialize TTS components based on config"""
        from tts.voice_utils import get_available_voices
        self.available_voices = get_available_voices()
        
        # Set default voice if none specified
        if self.config.voice_name is None:
            if self.available_voices:
                self.config.voice_name = self.available_voices[0]
            else:
                raise ValueError("No voices available. Please download voices first.")
        
    def speak_live(self, text: str) -> None:
        """Speak text using async generator for live playback"""
        from tts.audio_generator import AudioGenerator
        import threading
        
        if not text.strip():  # Skip empty text
            return
            
        generator = AudioGenerator(self.config.voice_name)
        event = threading.Event()
        
        try:
            # Add sentence to queue
            generator.add_sentence(text)
            
            # Process audio chunks as they become available
            while generator.is_processing() or not event.is_set():
                audio = generator.get_audio()
                if audio is not None and len(audio) > 0:
                    from tts.audio_player import play_audio
                    play_audio(audio, event, self.config.sample_rate)
                
        except Exception as e:
            print(f"Failed to generate/play audio: {str(e)}")
        finally:
            generator.stop()

    def speak(self, text: str, mode: str = "sync") -> None:
        """Speak text using specified mode (sync/live)"""
        if mode == "live":
            self.speak_live(text)
        else:  # sync mode
            from tts.audio_generator import AudioGeneratorSync
            from tts.audio_player import play_audio
            import threading
            
            if not text.strip():
                return
                
            generator = AudioGeneratorSync(
                voice_name=self.config.voice_name,
                speed=self.config.speed,
                volume=self.config.volume,
                sample_rate=self.config.sample_rate
            )
            
            try:
                audio = generator.add_sentence(text)
                if audio is not None and len(audio) > 0:
                    event = threading.Event()
                    play_audio(audio, event, self.config.sample_rate)
                else:
                    print(f"Warning: No audio generated for text: {text}")
            except Exception as e:
                print(f"Failed to generate/play audio: {str(e)}")

    def speak_async(self, text: str, callback: Optional[callable] = None) -> None:
        """Speak text asynchronously with optional callback"""
        from tts.audio_generator import AudioGenerator
        generator = AudioGenerator(self.config.voice_name)
        generator.add_sentence(text)
        # Implementation of async playback with callback

    def save_to_file(self, text: str, output_path: str) -> None:
        """Save speech to audio file"""
        from tts.audio_processor import generate_audio_from_file
        generate_audio_from_file(
            text, 
            self.config.voice_name,
            self.config.speed,
            output_path
        )
