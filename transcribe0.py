#!/usr/bin/env python3
"""
Transcribe0 - Local Audio Transcription Tool
Uses OpenAI Whisper for offline audio-to-text conversion
"""

import streamlit as st
import whisper
import tempfile
import os
from pathlib import Path
import time
from pydub import AudioSegment
from pydub.utils import make_chunks
import yt_dlp
import re

# Page configuration
st.set_page_config(
    page_title="Transcribe0",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="collapsed"
)

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
        <strong>Words:</strong> {:,}<br>
        <strong>Time saved:</strong> ~{:.0f} seconds
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

@st.cache_resource
def load_whisper_model(model_name):
    """Load and cache Whisper model"""
    with st.spinner(f'Loading {model_name} model... (first time may take a while)'):
        return whisper.load_model(model_name)

def is_youtube_url(url):
    """Check if the URL is a valid YouTube URL"""
    return YOUTUBE_REGEX.match(url) is not None

def get_youtube_info(url):
    """Get YouTube video info without downloading"""
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'extract_flat': False,
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        title = info.get('title', 'Unknown')
        duration = info.get('duration', 0)
        uploader = info.get('uploader', 'Unknown')
        return title, duration, uploader

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

def get_youtube_subtitles(url, language='auto'):
    """Get existing YouTube subtitles if available with intelligent language detection"""
    
    # First, get video info to detect the original language
    try:
        info_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
        }
        
        with yt_dlp.YoutubeDL(info_opts) as ydl:
            video_info = ydl.extract_info(url, download=False)
            video_language = video_info.get('language')
            automatic_captions = video_info.get('automatic_captions', {})
            subtitles = video_info.get('subtitles', {})
            
    except Exception:
        video_language = None
        automatic_captions = {}
        subtitles = {}
    
    # Determine subtitle languages to try based on detection
    if language == 'auto':
        # Build priority list based on available subtitles
        available_langs = set()
        
        # Check manual subtitles first (higher quality)
        if subtitles:
            available_langs.update(subtitles.keys())
        
        # Check automatic captions
        if automatic_captions:
            available_langs.update(automatic_captions.keys())
        
        # Create smart priority list
        if video_language and video_language in available_langs:
            # Prioritize detected video language
            subtitle_langs = [video_language]
            # Add other common languages as fallback
            for lang in ['es', 'en', 'fr', 'de', 'it', 'pt']:
                if lang != video_language and lang in available_langs:
                    subtitle_langs.append(lang)
        else:
            # Try Spanish first (since user mentioned Spanish video), then others
            priority_order = ['es', 'en', 'fr', 'de', 'it', 'pt']
            subtitle_langs = [lang for lang in priority_order if lang in available_langs]
            
        # If no matches, try all available
        if not subtitle_langs:
            subtitle_langs = list(available_langs)[:6]  # Limit to 6 languages
            
        # Always have English as final fallback
        if not subtitle_langs:
            subtitle_langs = ['es', 'en']
    else:
        # User specified language
        subtitle_langs = [language, 'en'] if language != 'en' else ['en']
    
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'writesubtitles': True,
        'writeautomaticsub': True,
        'subtitleslangs': subtitle_langs,
        'skip_download': True,
        'extract_flat': False,
    }
    
    with tempfile.TemporaryDirectory() as temp_dir:
        ydl_opts['outtmpl'] = os.path.join(temp_dir, '%(title)s.%(ext)s')
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.extract_info(url, download=True)
                
                # Look for subtitle files and prioritize by language
                subtitle_files = []
                for file in os.listdir(temp_dir):
                    if file.endswith('.vtt') or file.endswith('.srt'):
                        subtitle_files.append(file)
                
                if subtitle_files:
                    # More intelligent subtitle file selection
                    best_subtitle = None
                    detected_language = 'unknown'
                    
                    # First, try to find manual subtitles (higher quality)
                    manual_subtitle_files = [f for f in subtitle_files if not any(auto_indicator in f for auto_indicator in ['.auto.', 'auto-generated', 'a.'])]
                    auto_subtitle_files = [f for f in subtitle_files if f not in manual_subtitle_files]
                    
                    # Try manual subtitles first, then auto-generated
                    for file_group in [manual_subtitle_files, auto_subtitle_files]:
                        if not file_group:
                            continue
                            
                        # Try each language in priority order
                        for lang in subtitle_langs:
                            for file in file_group:
                                # Check for language in filename with various patterns
                                if (f'.{lang}.' in file or 
                                    f'-{lang}.' in file or 
                                    f'_{lang}.' in file or
                                    file.startswith(f'{lang}.') or
                                    f'.{lang}-' in file):
                                    best_subtitle = os.path.join(temp_dir, file)
                                    detected_language = lang
                                    break
                            if best_subtitle:
                                break
                        if best_subtitle:
                            break
                    
                    # If no language-specific file found, use the first manual subtitle, then first auto
                    if not best_subtitle:
                        preferred_files = manual_subtitle_files if manual_subtitle_files else auto_subtitle_files
                        if preferred_files:
                            best_subtitle = os.path.join(temp_dir, preferred_files[0])
                            # Try to detect language from filename with more patterns
                            filename = preferred_files[0]
                            for lang in subtitle_langs:
                                if (f'.{lang}.' in filename or 
                                    f'-{lang}.' in filename or 
                                    f'_{lang}.' in filename or
                                    filename.startswith(f'{lang}.') or
                                    f'.{lang}-' in filename):
                                    detected_language = lang
                                    break
                            
                            # If still unknown, analyze filename more
                            if detected_language == 'unknown':
                                # Look for common language patterns
                                if any(pattern in filename.lower() for pattern in ['spanish', 'español', 'es-']):
                                    detected_language = 'es'
                                elif any(pattern in filename.lower() for pattern in ['english', 'en-']):
                                    detected_language = 'en'
                                elif any(pattern in filename.lower() for pattern in ['french', 'français', 'fr-']):
                                    detected_language = 'fr'
                                else:
                                    # Use first language from priority list as best guess
                                    detected_language = subtitle_langs[0] if subtitle_langs else 'en'
                    
                    if best_subtitle:
                        with open(best_subtitle, 'r', encoding='utf-8') as f:
                            subtitle_content = f.read()
                        
                        # Clean VTT/SRT format to plain text
                        cleaned_text = clean_subtitle_text(subtitle_content)
                        return cleaned_text, True, detected_language
                
        except Exception:
            pass
    
    return None, False, 'none'

def download_youtube_audio(url):
    """Download audio from YouTube URL"""
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = os.path.join(temp_dir, '%(title)s.%(ext)s')
        
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': AudioConstants.DEFAULT_AUDIO_QUALITY,
            }],
            'outtmpl': output_path,
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                title = info.get('title', 'Unknown')
                duration = info.get('duration', 0)
                
                # List all files in temp directory for debugging
                all_files = os.listdir(temp_dir)
                
                # Find the downloaded file
                audio_file = None
                for file in all_files:
                    if file.endswith('.mp3'):
                        audio_file = os.path.join(temp_dir, file)
                        break
                
                if audio_file and os.path.exists(audio_file):
                    file_size = os.path.getsize(audio_file)
                    if file_size == 0:
                        raise Exception(f"Downloaded audio file is empty: {audio_file}")
                    
                    # Read the file content
                    with open(audio_file, 'rb') as f:
                        audio_content = f.read()
                    
                    if len(audio_content) == 0:
                        raise Exception("Audio content is empty after reading")
                        
                    return audio_content, title, duration
                else:
                    # Show available files for debugging
                    files_list = ", ".join(all_files) if all_files else "No files found"
                    raise Exception(f"No MP3 file found. Available files: {files_list}")
                    
        except Exception as e:
            # More detailed error information
            raise Exception(f"yt-dlp error: {str(e)}")

def get_audio_info(audio_path):
    """Get audio file information"""
    audio = AudioSegment.from_file(audio_path)
    duration = len(audio) / 1000  # Convert to seconds
    return {
        'duration': duration,
        'duration_str': f"{int(duration // 60)}:{int(duration % 60):02d}",
        'channels': audio.channels,
        'frame_rate': audio.frame_rate,
        'sample_width': audio.sample_width
    }

def process_large_audio(audio_path, model, language, chunk_duration_ms=AudioConstants.CHUNK_DURATION_MS):
    """Process large audio files in chunks (default 1 minute)"""
    audio = AudioSegment.from_file(audio_path)
    chunks = make_chunks(audio, chunk_duration_ms)
    
    transcriptions = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Container for real-time transcription display
    st.markdown("### 📝 Live Transcription")
    live_text = st.empty()
    
    for i, chunk in enumerate(chunks):
        # Update progress
        progress = (i + 1) / len(chunks)
        progress_bar.progress(progress)
        status_text.text(f'Processing chunk {i + 1} of {len(chunks)}...')
        
        # Export chunk to temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            chunk.export(tmp_file.name, format='wav')
            
            # Transcribe chunk
            if language == 'auto':
                result = model.transcribe(tmp_file.name)
            else:
                result = model.transcribe(tmp_file.name, language=language)
            
            transcriptions.append(result['text'])
            
            # Update live display with all transcribed text so far
            current_text = ' '.join(transcriptions)
            live_text.markdown(UIConstants.LIVE_TEXT_STYLE.format(current_text), unsafe_allow_html=True)
            
            # Clean up
            os.unlink(tmp_file.name)
    
    progress_bar.empty()
    status_text.empty()
    
    return ' '.join(transcriptions)

def transcribe_audio_file(audio_path, model_name, language, source_name="Audio"):
    """Transcribe audio from file path"""
    model = load_whisper_model(model_name)
    
    # Get audio info
    info = get_audio_info(audio_path)
    
    # Check if we need to chunk the audio (if longer than threshold)
    if info['duration'] > AudioConstants.LARGE_FILE_THRESHOLD:
        st.info(f"⏱️ Large file detected ({info['duration_str']}). Processing in chunks...")
        transcription = process_large_audio(audio_path, model, language)
    else:
        # Direct transcription for smaller files with live updates
        st.markdown("### 📝 Live Transcription")
        live_text = st.empty()
        segments_text = []
        
        with st.spinner(f'Transcribing {source_name} ({info["duration_str"]})...'):
            if language == 'auto':
                result = model.transcribe(audio_path, verbose=False)
            else:
                result = model.transcribe(audio_path, language=language, verbose=False)
            
            # Process segments for live display
            for segment in result['segments']:
                segments_text.append(segment['text'])
                current_text = ''.join(segments_text)
                live_text.markdown(UIConstants.LIVE_TEXT_STYLE.format(current_text), unsafe_allow_html=True)
            
            transcription = result['text']
    
    return transcription, info

def transcribe_audio(audio_file, model_name, language):
    """Main transcription function for uploaded files"""
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(audio_file.name).suffix) as tmp_file:
        tmp_file.write(audio_file.getvalue())
        tmp_path = tmp_file.name
    
    try:
        return transcribe_audio_file(tmp_path, model_name, language, audio_file.name)
    finally:
        # Clean up temporary file
        os.unlink(tmp_path)

@st.cache_data
def render_video_info_card(title, uploader, duration):
    """Render video information card with caching"""
    return UIConstants.VIDEO_INFO_STYLE.format(title, uploader, duration // 60, duration % 60)

@st.cache_data
def render_subtitle_info_card(detected_lang, word_count, estimated_time):
    """Render subtitle availability card with caching"""
    return UIConstants.SUBTITLE_AVAILABLE_STYLE.format(
        detected_lang.upper(), word_count, estimated_time
    )

def render_header():
    """Render application header"""
    st.markdown("""
    <h1 style='text-align: center; margin-bottom: 0;'>🎤 Transcribe0</h1>
    <p style='text-align: center; color: #666; margin-top: 0;'>Local audio-to-text transcription powered by OpenAI Whisper</p>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'transcribing' not in st.session_state:
        st.session_state['transcribing'] = False
    return st.session_state.get('transcribing', False)

def render_input_section(transcribing):
    """Handle file upload and YouTube URL input"""
    input_tab1, input_tab2 = st.tabs(["📁 Upload File", "🔗 YouTube URL"])
    
    uploaded_file = None
    youtube_url = None
    
    with input_tab1:
        uploaded_file = st.file_uploader(
            "Drag and drop or click to browse",
            type=ALL_FORMATS,
            help=f"Supported formats: {', '.join(ALL_FORMATS)}",
            label_visibility="collapsed",
            disabled=transcribing
        )
    
    with input_tab2:
        youtube_url = st.text_input(
            "Enter YouTube URL",
            placeholder="https://www.youtube.com/watch?v=...",
            help="Paste a YouTube video URL to transcribe its audio",
            label_visibility="collapsed",
            disabled=transcribing
        )
        st.caption("ℹ️ Audio is downloaded temporarily and deleted after transcription")
        
        if youtube_url and is_youtube_url(youtube_url):
            st.markdown('<p style="color: #6B7280;">✅ Valid YouTube URL</p>', unsafe_allow_html=True)
            
            # Cache video info in session state by URL
            cache_key = f"video_info_{hash(youtube_url)}"
            if cache_key not in st.session_state:
                try:
                    with st.spinner('🔍 Getting video info...'):
                        title, duration, uploader = get_youtube_info(youtube_url)
                        subtitle_text, has_subtitles, detected_lang = get_youtube_subtitles(youtube_url, 'auto')
                    
                    # Cache all info together
                    st.session_state[cache_key] = {
                        'title': title,
                        'duration': duration,
                        'uploader': uploader,
                        'subtitle_text': subtitle_text,
                        'has_subtitles': has_subtitles,
                        'detected_lang': detected_lang
                    }
                except Exception as e:
                    st.warning(f"⚠️ Could not get video info: {str(e)}")
                    st.session_state[cache_key] = None
            
            # Use cached data
            if st.session_state[cache_key]:
                cached_data = st.session_state[cache_key]
                title = cached_data['title']
                duration = cached_data['duration']
                uploader = cached_data['uploader']
                has_subtitles = cached_data['has_subtitles']
                detected_lang = cached_data['detected_lang']
                
                # Only show video info card when not transcribing
                if not transcribing:
                    st.markdown(render_video_info_card(title, uploader, duration), unsafe_allow_html=True)
                
                if has_subtitles:
                    word_count = len(cached_data['subtitle_text'].split())
                    estimated_whisper_time = duration * 0.1
                    
                    st.markdown(
                        render_subtitle_info_card(detected_lang, word_count, estimated_whisper_time),
                        unsafe_allow_html=True
                    )
                    
                    # Only auto-store YouTube subtitles if we haven't manually triggered Whisper
                    if not st.session_state.get('manual_whisper_requested', False):
                        st.session_state['transcription'] = cached_data['subtitle_text']
                        st.session_state['audio_info'] = {
                            'duration': duration,
                            'duration_str': f"{int(duration // 60)}:{int(duration % 60):02d}",
                            'channels': 2,
                            'frame_rate': 44100,
                            'sample_width': 2
                        }
                        st.session_state['processing_time'] = 0.1  # Instant
                        st.session_state['source_name'] = title
                        st.session_state['transcription_source'] = f'YouTube Transcription ({detected_lang.upper()})'
                        st.session_state['youtube_subtitles_shown'] = True
                else:
                    st.markdown(UIConstants.NO_SUBTITLES_STYLE, unsafe_allow_html=True)
                    
        elif youtube_url:
            st.error("❌ Invalid YouTube URL. Please enter a valid YouTube link.")
    
    return uploaded_file, youtube_url

def render_settings_section(transcribing):
    """Handle model and language settings"""
    st.markdown("### ⚙️ Settings")
    
    if transcribing:
        # Show locked settings
        locked_model = st.session_state.get('locked_model', 'large-v3-turbo')
        locked_language = st.session_state.get('locked_language', 'auto')
        locked_use_youtube = st.session_state.get('locked_use_youtube', False)
        
        if locked_use_youtube:
            st.markdown("**Source**")
            st.markdown("🔒 YouTube Subtitles")
            st.markdown("<br>", unsafe_allow_html=True)
        else:
            st.markdown("**Model Quality**")
            st.markdown(f"🔒 {locked_model.capitalize()} - {MODELS[locked_model]}")
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**Language**")
            st.markdown(f"🔒 {LANGUAGES[locked_language]}")
        
        st.warning("⚠️ Settings locked during transcription")
        
        model_name = locked_model
        language = locked_language
    else:
        # Always show Whisper settings (YouTube subtitles are auto-shown when available)
        st.markdown("**Model Quality**")
        model_name = st.selectbox(
            "Model",
            options=list(MODELS.keys()),
            index=5,
            format_func=lambda x: f"{x.capitalize()} - {MODELS[x]}",
            label_visibility="collapsed"
        )
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Language**")
        language = st.selectbox(
            "Language",
            options=list(LANGUAGES.keys()),
            index=0,
            format_func=lambda x: LANGUAGES[x],
            label_visibility="collapsed"
        )
    
    return model_name, language
def determine_audio_source(uploaded_file, youtube_url):
    """Determine the audio source and display relevant information"""
    audio_source = None
    source_name = None
    
    if uploaded_file is not None:
        audio_source = uploaded_file
        source_name = uploaded_file.name
        file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
        st.markdown(f"""
        <div style='padding: 1rem; background-color: #1a1a1a; border-radius: 0.5rem; margin: 1rem 0;'>
            <strong>📁 File:</strong> {uploaded_file.name}<br>
            <strong>📊 Size:</strong> {file_size_mb:.1f} MB
        </div>
        """, unsafe_allow_html=True)
    elif youtube_url and is_youtube_url(youtube_url):
        audio_source = youtube_url
        # Get the video title for source_name (use cached data)
        cache_key = f"video_info_{hash(youtube_url)}"
        if cache_key in st.session_state and st.session_state[cache_key]:
            source_name = st.session_state[cache_key]['title']
        else:
            source_name = "YouTube Video"
    
    return audio_source, source_name
def handle_transcription_process(audio_source, source_name, model_name, language, transcribing):
    """Handle the transcription workflow"""
    if audio_source is None:
        return
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Check if YouTube subtitles are already shown
    youtube_subtitles_already_shown = st.session_state.get('youtube_subtitles_shown', False)
    
    if not transcribing and not youtube_subtitles_already_shown:
        if st.button("🎯 Start Transcription", type="primary", use_container_width=True):
            # Lock settings and start transcription
            st.session_state['transcribing'] = True
            st.session_state['locked_model'] = model_name
            st.session_state['locked_language'] = language
            st.session_state['locked_use_youtube'] = False  # Always use Whisper when manually triggered
            st.rerun()
    elif youtube_subtitles_already_shown:
        st.markdown("""
        <div style='padding: 1rem; background-color: #2a2a3a; border-radius: 0.5rem; margin: 1rem 0; border-left: 4px solid #6B7280;'>
            <strong>YouTube transcription is already <a href="#transcription-results" style="color: #9CA3AF; text-decoration: underline;">displayed below</a>!</strong><br>
            Use the "Regenerate with Whisper" button if you want generate transcription again.
        </div>
        """, unsafe_allow_html=True)
        
        # Show regenerate button right after the message
        if st.button("Regenerate with Whisper", type="secondary", use_container_width=True):
            # Set flag to indicate manual Whisper request
            st.session_state['manual_whisper_requested'] = True
            st.session_state['youtube_subtitles_shown'] = False
            st.session_state['transcribing'] = True
            st.session_state['locked_model'] = 'large-v3-turbo'
            st.session_state['locked_language'] = 'auto'
            st.session_state['locked_use_youtube'] = False
            
            # Preserve YouTube transcription data before clearing
            if 'transcription' in st.session_state:
                st.session_state['youtube_transcription'] = st.session_state['transcription']
                st.session_state['youtube_source'] = st.session_state.get('transcription_source', 'YouTube Transcription')
                st.session_state['youtube_audio_info'] = st.session_state.get('audio_info', {})
                st.session_state['youtube_processing_time'] = st.session_state.get('processing_time', 0.1)
            
            # Clear current transcription to show fresh interface
            if 'transcription' in st.session_state:
                del st.session_state['transcription']
            if 'transcription_source' in st.session_state:
                del st.session_state['transcription_source']
            if 'audio_info' in st.session_state:
                del st.session_state['audio_info']
            if 'processing_time' in st.session_state:
                del st.session_state['processing_time']
                
            st.rerun()
    else:
        # Show progress and perform transcription
        try:
            start_time = time.time()
            
            # Handle YouTube URL vs file upload (always use Whisper when manually triggered)
            if isinstance(audio_source, str):  # YouTube URL
                # Use Whisper transcription for YouTube
                with st.status("🔄 Processing YouTube video with Whisper...", expanded=True) as status:
                    st.write("📥 Temporarily downloading audio from YouTube...")
                    
                    try:
                        audio_content, video_title, duration = download_youtube_audio(audio_source)
                        audio_size_mb = len(audio_content) / (1024 * 1024)
                        
                        st.write(f"✅ Audio downloaded ({audio_size_mb:.1f} MB)")
                        
                    except Exception as e:
                        st.error(f"❌ Failed to download audio: {str(e)}")
                        st.session_state['transcribing'] = False
                        st.rerun()
                        return
                    
                    # Save audio to temporary file
                    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_file:
                        tmp_file.write(audio_content)
                        temp_audio_path = tmp_file.name
                    
                    st.write("🎯 Starting Whisper transcription...")
                    status.update(label="🎯 Transcribing with Whisper...", state="running")
                
                    try:
                        transcription, audio_info = transcribe_audio_file(
                            temp_audio_path, 
                            model_name, 
                            language if language != 'auto' else None,
                            video_title
                        )
                        status.update(label="✅ Transcription completed!", state="complete")
                    finally:
                        # Always clean up the temporary file
                        if os.path.exists(temp_audio_path):
                            os.unlink(temp_audio_path)
                
            else:  # File upload
                transcription, audio_info = transcribe_audio(
                    audio_source, 
                    model_name, 
                    language if language != 'auto' else None
                )
            
            # Store results and reset state
            processing_time = time.time() - start_time
            
            # If this is a regeneration from YouTube, store both transcriptions
            if st.session_state.get('manual_whisper_requested', False):
                # Store Whisper transcription separately
                st.session_state['whisper_transcription'] = transcription
                st.session_state['whisper_audio_info'] = audio_info
                st.session_state['whisper_processing_time'] = processing_time
                st.session_state['whisper_source'] = f'OpenAI Whisper ({model_name})'
                st.session_state['has_both_transcriptions'] = True
                
                # Keep current transcription as active (now Whisper)
                st.session_state['transcription'] = transcription
                st.session_state['audio_info'] = audio_info
                st.session_state['processing_time'] = processing_time
                st.session_state['transcription_source'] = f'OpenAI Whisper ({model_name})'
            else:
                # Normal single transcription
                st.session_state['transcription'] = transcription
                st.session_state['audio_info'] = audio_info
                st.session_state['processing_time'] = processing_time
                st.session_state['transcription_source'] = f'OpenAI Whisper ({model_name})'
            
            st.session_state['source_name'] = source_name
            st.session_state['transcribing'] = False
            
            # Clear manual request flag after completion
            if 'manual_whisper_requested' in st.session_state:
                del st.session_state['manual_whisper_requested']
            
            st.success(f"✅ Transcription completed in {processing_time:.1f} seconds!")
            st.rerun()
            
        except Exception as e:
            st.session_state['transcribing'] = False
            # Clear manual request flag if there's an error
            if 'manual_whisper_requested' in st.session_state:
                del st.session_state['manual_whisper_requested']
            st.error(f"❌ Error during transcription: {str(e)}")
            st.info("Please check if the file format is supported and try again.")
            st.rerun()
def render_results_section():
    """Display transcription results and export options"""
    if 'transcription' not in st.session_state:
        return
        
    st.markdown("<hr style='margin: 2rem 0; opacity: 0.2;'>", unsafe_allow_html=True)
    st.markdown('<h3 id="transcription-results">📝 Transcription Results</h3>', unsafe_allow_html=True)
    
    # Check if we have both transcriptions
    has_both = st.session_state.get('has_both_transcriptions', False)
    
    if has_both:
        # Show both transcriptions in tabs
        tab1, tab2 = st.tabs(["Whisper Transcription", "YouTube Transcription"])
        
        with tab1:
            render_single_transcription(
                transcription=st.session_state['whisper_transcription'],
                source=st.session_state['whisper_source'],
                audio_info=st.session_state['whisper_audio_info'],
                processing_time=st.session_state['whisper_processing_time'],
                tab_key="whisper"
            )
        
        with tab2:
            render_single_transcription(
                transcription=st.session_state['youtube_transcription'],
                source=st.session_state['youtube_source'],
                audio_info=st.session_state['youtube_audio_info'],
                processing_time=st.session_state['youtube_processing_time'],
                tab_key="youtube"
            )
    else:
        # Show single transcription
        render_single_transcription(
            transcription=st.session_state['transcription'],
            source=st.session_state.get('transcription_source', ''),
            audio_info=st.session_state['audio_info'],
            processing_time=st.session_state['processing_time'],
            tab_key="single"
        )

def render_single_transcription(transcription, source, audio_info, processing_time, tab_key):
    """Render a single transcription with metrics and export options"""
    # Show transcription source
    if source:
        if 'YouTube Transcription' in source:
            st.markdown(f"""
            <div style='padding: 0.5rem 1rem; background-color: #2a2a3a; border-radius: 0.5rem; margin: 0.5rem 0; border-left: 4px solid #6B7280;'>
                <strong>Source:</strong> {source} • <strong>Instant Result</strong>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style='padding: 0.5rem 1rem; background-color: #2a2a3a; border-radius: 0.5rem; margin: 0.5rem 0; border-left: 4px solid #6B7280;'>
                <strong>Source:</strong> {source}
            </div>
            """, unsafe_allow_html=True)
    
    # Show metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Duration", audio_info['duration_str'])
    with col2:
        if 'YouTube Transcription' in source:
            st.metric("Processing Time", "⚡ Instant")
        else:
            st.metric("Processing Time", f"{processing_time:.1f}s")
    
    # Show transcription text
    st.markdown("**Transcribed Text**")
    transcribed_text = st.text_area(
        "Transcribed Text",
        value=transcription,
        height=400,
        help="You can edit the text before copying or downloading",
        label_visibility="collapsed",
        key=f"transcription_output_{tab_key}"
    )
    
    # Export options
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🛠️ Export Options")
    
    action_col1, action_col2, action_col3 = st.columns([1, 1, 1])
    
    with action_col1:
        if st.button("📋 Copy to Clipboard", use_container_width=True, key=f"copy_{tab_key}"):
            st.code(transcribed_text, language=None)
            st.success("✅ Text ready to copy! Use the copy button in the code block above.")
    
    with action_col2:
        source_name = st.session_state.get('source_name', 'transcription')
        clean_name = "".join(c for c in source_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        
        # Add source type to filename
        source_type = "whisper" if "Whisper" in source else "youtube"
        filename = f"{clean_name}_{source_type}_{int(time.time())}.txt" if clean_name else f"transcription_{source_type}_{int(time.time())}.txt"
        
        st.download_button(
            label="💾 Download as .txt",
            data=transcribed_text,
            file_name=filename,
            mime="text/plain",
            use_container_width=True,
            key=f"download_{tab_key}"
        )
    
    with action_col3:
        word_count = len(transcribed_text.split())
        char_count = len(transcribed_text)
        st.metric("Word Count", f"{word_count:,}", help=f"{char_count:,} characters")
def render_footer():
    """Render application footer"""
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; padding: 2rem 0; opacity: 0.6; font-size: 0.9rem;'>
        🔒 All processing happens locally on your machine<br>
        No data is sent to external servers
    </div>
    """, unsafe_allow_html=True)

def main():
    """Main application function - now optimized and modular"""
    render_header()
    
    # Initialize session state
    transcribing = initialize_session_state()
    
    # Create columns for layout with better spacing
    col1, col2 = st.columns([2.5, 1], gap="large")
    
    # Input section
    with col1:
        uploaded_file, youtube_url = render_input_section(transcribing)
    
    # Settings section
    with col2:
        model_name, language = render_settings_section(transcribing)
    
    # Determine audio source
    audio_source, source_name = determine_audio_source(uploaded_file, youtube_url)
    
    # Transcription process
    handle_transcription_process(audio_source, source_name, model_name, language, transcribing)
    
    # Display results
    render_results_section()
    
    # Footer
    render_footer()

if __name__ == "__main__":
    main()