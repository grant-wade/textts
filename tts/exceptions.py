class TTSException(Exception):
    """Base exception for TTS operations"""
    pass

class ConfigurationError(TTSException):
    """Invalid configuration"""
    pass

class EngineInitializationError(TTSException):
    """Failed to initialize TTS engine"""
    pass

class SpeechGenerationError(TTSException):
    """Failed to generate speech"""
    pass

class AudioPlaybackError(TTSException):
    """Failed to play audio"""
    pass
