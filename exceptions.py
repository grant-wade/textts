class TTSError(Exception):
    """Base exception class for TTS errors"""
    pass

class VoiceNotFoundError(TTSError):
    """Raised when specified voice is not available"""
    pass

class InputFileError(TTSError):
    """Raised when there are issues with the input file"""
    pass

class AudioProcessingError(TTSError):
    """Raised when audio processing fails"""
    pass

class AudioPlaybackError(TTSError):
    """Raised when audio playback fails"""
    pass

class StdinProcessingError(TTSError):
    """Raised when processing stdin input fails"""
    pass
