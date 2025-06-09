"""
AudioInk - Constants and Configuration
"""

import re

# Supported formats
AUDIO_FORMATS = ['mp3', 'wav', 'm4a', 'flac', 'ogg']
VIDEO_FORMATS = ['mp4', 'avi', 'mov']
ALL_FORMATS = AUDIO_FORMATS + VIDEO_FORMATS

# Model options
MODELS = {
    'tiny': 'Ultra fast, lower quality',
    'base': 'Balanced speed and quality',
    'small': 'Good quality, moderate speed',
    'medium': 'High quality, slower',
    'large': 'Best quality, slowest',
    'large-v3-turbo': 'Excellent quality, faster than large'
}

# Language options
LANGUAGES = {
    'auto': 'Auto-detect',
    'en': 'English',
    'es': 'Spanish',
    'fr': 'French',
    'de': 'German',
    'it': 'Italian',
    'pt': 'Portuguese'
}

# Pre-compiled regex patterns for performance
YOUTUBE_REGEX = re.compile(r'(https?://)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)/(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})')
TIMESTAMP_REGEX = re.compile(r'<\d{2}:\d{2}:\d{2}\.\d{3}>')
HTML_TAG_REGEX = re.compile(r'<[^>]+>')
WHITESPACE_REGEX = re.compile(r'\s+')
BRACKET_CONTENT_REGEX = re.compile(r'\[.*?\]')
PAREN_CONTENT_REGEX = re.compile(r'\(.*?\)')
TIME_PATTERN_REGEX = re.compile(r'^\d{2}:\d{2}:\d{2}')

# UI Constants
class UIConstants:
    LIVE_TEXT_STYLE = """
    <div style='padding: 1rem; background-color: #1a1a1a; border-radius: 0.5rem; 
                height: 300px; overflow-y: auto; font-family: monospace;'>
        {}
    </div>
    """
    
    VIDEO_INFO_STYLE = """
    <div style='padding: 1rem; background-color: #1a1a1a; border-radius: 0.5rem; margin: 1rem 0;'>
        <strong>Video:</strong> {}<br>
        <strong>Channel:</strong> {}<br>
        <strong>Duration:</strong> {}:{:02d}
    </div>
    """
    
    SUBTITLE_AVAILABLE_STYLE = """
    <div style='padding: 1rem; background-color: #2a2a3a; border-radius: 0.5rem; margin: 1rem 0; border-left: 4px solid #6B7280;'>
        <strong>⚡ YouTube Transcription Available</strong><br>
        <strong>Language:</strong> {}<br>
        <strong>Words:</strong> {:,}
    </div>
    """
    
    NO_SUBTITLES_STYLE = """
    <div style='padding: 1rem; background-color: #4a4a1a; border-radius: 0.5rem; margin: 1rem 0; border-left: 4px solid #FFC107;'>
        <strong>⚠️ No YouTube subtitles found</strong><br>
        Will use Whisper transcription instead
    </div>
    """

# Audio processing constants
class AudioConstants:
    CHUNK_DURATION_MS = 60000
    LARGE_FILE_THRESHOLD = 120  # seconds
    DEFAULT_AUDIO_QUALITY = '192'