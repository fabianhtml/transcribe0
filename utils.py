"""
AudioInk - Utilities Module
Helper functions and text processing utilities
"""

import streamlit as st
from constants import (
    TIMESTAMP_REGEX, HTML_TAG_REGEX, WHITESPACE_REGEX, 
    BRACKET_CONTENT_REGEX, PAREN_CONTENT_REGEX, TIME_PATTERN_REGEX
)

def clean_subtitle_text(subtitle_content):
    """Clean subtitle text from various formats (VTT, SRT, YouTube) - optimized version"""
    # Apply regex substitutions using pre-compiled patterns
    text = TIMESTAMP_REGEX.sub('', subtitle_content)
    text = HTML_TAG_REGEX.sub('', text)
    
    # Process lines more efficiently
    lines = text.split('\n')
    clean_lines = []
    seen_sentences = set()
    
    # Pre-define skip conditions for better performance
    skip_prefixes = ('WEBVTT', 'Kind:', 'Language:', 'NOTE')
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines or very short lines
        if len(line) < 3:
            continue
            
        # Skip VTT headers and NOTE lines
        if line.startswith(skip_prefixes):
            continue
            
        # Skip timestamp lines and pure numbers
        if '-->' in line or line.isdigit():
            continue
            
        # Skip timestamp patterns at start of line
        if TIME_PATTERN_REGEX.match(line):
            continue
            
        # Clean up the line - normalize whitespace
        line = ' '.join(line.split())
        
        # Add to clean lines if not seen before (removes duplicates)
        line_key = line.lower().strip('.,!?')
        if line_key not in seen_sentences and len(line_key) > 5:
            seen_sentences.add(line_key)
            clean_lines.append(line)
    
    # Join all clean lines
    result = ' '.join(clean_lines)
    
    # Final cleanup using pre-compiled patterns
    result = WHITESPACE_REGEX.sub(' ', result)
    result = BRACKET_CONTENT_REGEX.sub('', result)  # Remove [Music], [Applause], etc.
    result = PAREN_CONTENT_REGEX.sub('', result)    # Remove (inaudible), etc.
    
    return result.strip()

def has_youtube_transcription():
    """Check if there's an active YouTube transcription"""
    return ('transcription' in st.session_state and 
            'YouTube' in str(st.session_state.get('transcription_source', '')))

def has_file_transcription():
    """Check if there's an active file transcription"""
    return ('transcription' in st.session_state and 
            'Whisper' in str(st.session_state.get('transcription_source', '')))

def clear_youtube_data():
    """Clear all YouTube-related data from session state"""
    youtube_keys_to_clear = []
    for key in list(st.session_state.keys()):
        if (key.startswith('video_info_') or 
            key == 'youtube_url_field' or
            ('YouTube' in str(st.session_state.get('transcription_source', '')) and 
             key in ['transcription', 'transcription_source', 'audio_info', 'processing_time', 
                    'source_name', 'youtube_subtitles_shown'])):
            youtube_keys_to_clear.append(key)
    
    for key in youtube_keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

def clear_file_data():
    """Clear all file upload-related data from session state"""
    file_keys_to_clear = []
    for key in list(st.session_state.keys()):
        if (key == 'file_upload_widget' or
            ('Whisper' in str(st.session_state.get('transcription_source', '')) and 
             key in ['transcription', 'transcription_source', 'audio_info', 'processing_time', 
                    'source_name'])):
            file_keys_to_clear.append(key)
    
    for key in file_keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]