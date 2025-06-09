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
import io
import numpy as np
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

@st.cache_resource
def load_whisper_model(model_name):
    """Load and cache Whisper model"""
    with st.spinner(f'Loading {model_name} model... (first time may take a while)'):
        return whisper.load_model(model_name)

def is_youtube_url(url):
    """Check if the URL is a valid YouTube URL"""
    youtube_regex = r'(https?://)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)/(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})'
    return re.match(youtube_regex, url) is not None

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

def download_youtube_audio(url):
    """Download audio from YouTube URL"""
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = os.path.join(temp_dir, '%(title)s.%(ext)s')
        
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': output_path,
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            title = info.get('title', 'Unknown')
            duration = info.get('duration', 0)
            
            # Find the downloaded file
            audio_file = None
            for file in os.listdir(temp_dir):
                if file.endswith('.mp3'):
                    audio_file = os.path.join(temp_dir, file)
                    break
            
            if audio_file and os.path.exists(audio_file):
                # Read the file content
                with open(audio_file, 'rb') as f:
                    audio_content = f.read()
                return audio_content, title, duration
            else:
                raise Exception("Failed to download audio")

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

def process_large_audio(audio_path, model, language, chunk_duration_ms=60000):
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
            live_text.markdown(f"""
            <div style='padding: 1rem; background-color: #1a1a1a; border-radius: 0.5rem; 
                        height: 300px; overflow-y: auto; font-family: monospace;'>
                {current_text}
            </div>
            """, unsafe_allow_html=True)
            
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
    
    # Check if we need to chunk the audio (if longer than 2 minutes)
    if info['duration'] > 120:
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
                live_text.markdown(f"""
                <div style='padding: 1rem; background-color: #1a1a1a; border-radius: 0.5rem; 
                            height: 300px; overflow-y: auto; font-family: monospace;'>
                    {current_text}
                </div>
                """, unsafe_allow_html=True)
            
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

def main():
    # Header with custom styling
    st.markdown("""
    <h1 style='text-align: center; margin-bottom: 0;'>🎤 Transcribe0</h1>
    <p style='text-align: center; color: #666; margin-top: 0;'>Local audio-to-text transcription powered by OpenAI Whisper</p>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Initialize session state
    if 'transcribing' not in st.session_state:
        st.session_state['transcribing'] = False
    
    transcribing = st.session_state.get('transcribing', False)
    
    # Create columns for layout with better spacing
    col1, col2 = st.columns([2.5, 1], gap="large")
    
    # Input section
    with col1:
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
                try:
                    with st.spinner('🔍 Getting video info...'):
                        title, duration, uploader = get_youtube_info(youtube_url)
                    
                    st.markdown(f"""
                    <div style='padding: 1rem; background-color: #1a1a1a; border-radius: 0.5rem; margin: 1rem 0;'>
                        <strong>Video:</strong> {title}<br>
                        <strong>Channel:</strong> {uploader}<br>
                        <strong>Duration:</strong> {duration // 60}:{duration % 60:02d}
                    </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.warning(f"⚠️ Could not get video info: {str(e)}")
            elif youtube_url:
                st.error("❌ Invalid YouTube URL. Please enter a valid YouTube link.")
    
    # Settings section
    with col2:
        st.markdown("### ⚙️ Settings")
        
        if transcribing:
            # Show locked settings
            locked_model = st.session_state.get('locked_model', 'large-v3-turbo')
            locked_language = st.session_state.get('locked_language', 'auto')
            
            st.markdown("**Model Quality**")
            st.markdown(f"🔒 {locked_model.capitalize()} - {MODELS[locked_model]}")
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**Language**")
            st.markdown(f"🔒 {LANGUAGES[locked_language]}")
            st.warning("⚠️ Settings locked during transcription")
            
            model_name = locked_model
            language = locked_language
        else:
            # Normal settings
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
    
    # Determine audio source
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
        try:
            title, _, _ = get_youtube_info(youtube_url)
            source_name = title
        except:
            source_name = "YouTube Video"
    
    # Transcription button and process
    if audio_source is not None:
        st.markdown("<br>", unsafe_allow_html=True)
        
        if not transcribing:
            if st.button("🎯 Start Transcription", type="primary", use_container_width=True):
                # Lock settings and start transcription
                st.session_state['transcribing'] = True
                st.session_state['locked_model'] = model_name
                st.session_state['locked_language'] = language
                st.rerun()
        else:
            # Show progress and perform transcription
            st.info("🔄 Transcription in progress...")
            
            try:
                start_time = time.time()
                
                # Handle YouTube URL vs file upload
                if isinstance(audio_source, str):  # YouTube URL
                    with st.spinner('📥 Downloading audio from YouTube...'):
                        audio_content, video_title, duration = download_youtube_audio(audio_source)
                        
                    st.markdown(f"""
                    <div style='padding: 1rem; background-color: #1a1a1a; border-radius: 0.5rem; margin: 1rem 0;'>
                        <strong>Video:</strong> {video_title}<br>
                        <strong>Duration:</strong> {duration // 60}:{duration % 60:02d}<br>
                        <strong>Status:</strong> Downloaded successfully
                    </div>
                    """, unsafe_allow_html=True)
                    
                    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_file:
                        tmp_file.write(audio_content)
                        temp_audio_path = tmp_file.name
                    
                    transcription, audio_info = transcribe_audio_file(
                        temp_audio_path, 
                        model_name, 
                        language if language != 'auto' else None,
                        video_title
                    )
                    
                    os.unlink(temp_audio_path)
                    
                else:  # File upload
                    transcription, audio_info = transcribe_audio(
                        audio_source, 
                        model_name, 
                        language if language != 'auto' else None
                    )
                
                # Store results and reset state
                processing_time = time.time() - start_time
                st.session_state['transcription'] = transcription
                st.session_state['audio_info'] = audio_info
                st.session_state['processing_time'] = processing_time
                st.session_state['source_name'] = source_name
                st.session_state['transcribing'] = False
                
                st.success(f"✅ Transcription completed in {processing_time:.1f} seconds!")
                st.rerun()
                
            except Exception as e:
                st.session_state['transcribing'] = False
                st.error(f"❌ Error during transcription: {str(e)}")
                st.info("Please check if the file format is supported and try again.")
                st.rerun()
    
    # Display results
    if 'transcription' in st.session_state:
        st.markdown("<hr style='margin: 2rem 0; opacity: 0.2;'>", unsafe_allow_html=True)
        st.markdown("### 📝 Transcription Results")
        
        info = st.session_state['audio_info']
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Duration", info['duration_str'])
        with col2:
            st.metric("Processing Time", f"{st.session_state['processing_time']:.1f}s")
        with col3:
            st.metric("Speed", f"{info['duration'] / st.session_state['processing_time']:.1f}x")
        
        st.markdown("**Transcribed Text**")
        transcribed_text = st.text_area(
            "Transcribed Text",
            value=st.session_state['transcription'],
            height=500,
            help="You can edit the text before copying or downloading",
            label_visibility="collapsed",
            key="transcription_output"
        )
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 🛠️ Export Options")
        
        action_col1, action_col2, action_col3 = st.columns([1, 1, 1])
        
        with action_col1:
            if st.button("📋 Copy to Clipboard", use_container_width=True):
                st.code(transcribed_text, language=None)
                st.success("✅ Text ready to copy! Use the copy button in the code block above.")
        
        with action_col2:
            source_name = st.session_state.get('source_name', 'transcription')
            clean_name = "".join(c for c in source_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
            filename = f"{clean_name}_{int(time.time())}.txt" if clean_name else f"transcription_{int(time.time())}.txt"
            
            st.download_button(
                label="💾 Download as .txt",
                data=transcribed_text,
                file_name=filename,
                mime="text/plain",
                use_container_width=True
            )
        
        with action_col3:
            word_count = len(transcribed_text.split())
            char_count = len(transcribed_text)
            st.metric("Word Count", f"{word_count:,}", help=f"{char_count:,} characters")
    
    # Footer
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; padding: 2rem 0; opacity: 0.6; font-size: 0.9rem;'>
        🔒 All processing happens locally on your machine<br>
        No data is sent to external servers
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()