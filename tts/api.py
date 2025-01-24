from typing import Optional
from .engine import TTSEngine, TTSConfig

class TTS:
    @staticmethod
    def create(config: Optional[TTSConfig] = None) -> TTSEngine:
        """Create a new TTS engine instance"""
        return TTSEngine(config)
    
    @staticmethod
    def speak(text: str, voice: Optional[str] = None) -> None:
        """Quick access to speak text"""
        config = TTSConfig(voice_name=voice)
        engine = TTSEngine(config)
        engine.speak(text)
