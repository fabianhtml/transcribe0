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
    'large': 'Best quality, slowest'
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
                        height: 200px; overflow-y: auto; font-family: monospace;'>
                {current_text}
            </div>
            """, unsafe_allow_html=True)
            
            # Clean up
            os.unlink(tmp_file.name)
    
    progress_bar.empty()
    status_text.empty()
    
    return ' '.join(transcriptions)

def transcribe_audio(audio_file, model_name, language):
    """Main transcription function"""
    model = load_whisper_model(model_name)
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(audio_file.name).suffix) as tmp_file:
        tmp_file.write(audio_file.getvalue())
        tmp_path = tmp_file.name
    
    try:
        # Get audio info
        info = get_audio_info(tmp_path)
        
        # Check if we need to chunk the audio (if longer than 2 minutes)
        if info['duration'] > 120:
            st.info(f"⏱️ Large file detected ({info['duration_str']}). Processing in chunks...")
            transcription = process_large_audio(tmp_path, model, language)
        else:
            # Direct transcription for smaller files with live updates
            st.markdown("### 📝 Live Transcription")
            live_text = st.empty()
            segments_text = []
            
            with st.spinner(f'Transcribing audio ({info["duration_str"]})...'):
                if language == 'auto':
                    result = model.transcribe(tmp_path, verbose=False)
                else:
                    result = model.transcribe(tmp_path, language=language, verbose=False)
                
                # Process segments for live display
                for segment in result['segments']:
                    segments_text.append(segment['text'])
                    current_text = ''.join(segments_text)
                    live_text.markdown(f"""
                    <div style='padding: 1rem; background-color: #1a1a1a; border-radius: 0.5rem; 
                                height: 200px; overflow-y: auto; font-family: monospace;'>
                        {current_text}
                    </div>
                    """, unsafe_allow_html=True)
                
                transcription = result['text']
        
        return transcription, info
        
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
    
    # Create columns for layout with better spacing
    col1, col2 = st.columns([2.5, 1], gap="large")
    
    with col1:
        # File uploader with custom styling
        st.markdown("### 📁 Upload Audio/Video")
        uploaded_file = st.file_uploader(
            "Drag and drop or click to browse",
            type=ALL_FORMATS,
            help=f"Supported formats: {', '.join(ALL_FORMATS)}",
            label_visibility="collapsed"
        )
    
    with col2:
        st.markdown("### ⚙️ Settings")
        
        # Model selection with better formatting
        st.markdown("**Model Quality**")
        model_name = st.selectbox(
            "Model",
            options=list(MODELS.keys()),
            index=1,  # Default to 'base'
            format_func=lambda x: f"{x.capitalize()} - {MODELS[x]}",
            label_visibility="collapsed"
        )
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Language selection
        st.markdown("**Language**")
        language = st.selectbox(
            "Language",
            options=list(LANGUAGES.keys()),
            index=0,  # Default to 'auto'
            format_func=lambda x: LANGUAGES[x],
            label_visibility="collapsed"
        )
    
    # Process audio if file is uploaded
    if uploaded_file is not None:
        # File info with better styling
        file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
        st.markdown(f"""
        <div style='padding: 1rem; background-color: #1a1a1a; border-radius: 0.5rem; margin: 1rem 0;'>
            <strong>📁 File:</strong> {uploaded_file.name}<br>
            <strong>📊 Size:</strong> {file_size_mb:.1f} MB
        </div>
        """, unsafe_allow_html=True)
        
        # Transcribe button with better styling
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🎯 Start Transcription", type="primary", use_container_width=True):
            try:
                # Store in session state
                start_time = time.time()
                
                # Perform transcription
                transcription, audio_info = transcribe_audio(
                    uploaded_file, 
                    model_name, 
                    language if language != 'auto' else None
                )
                
                # Calculate processing time
                processing_time = time.time() - start_time
                
                # Store results in session state
                st.session_state['transcription'] = transcription
                st.session_state['audio_info'] = audio_info
                st.session_state['processing_time'] = processing_time
                
                st.success(f"✅ Transcription completed in {processing_time:.1f} seconds!")
                
            except Exception as e:
                st.error(f"❌ Error during transcription: {str(e)}")
                st.info("Please check if the file format is supported and try again.")
    
    # Display results if available
    if 'transcription' in st.session_state:
        # Results section with better styling
        st.markdown("<hr style='margin: 2rem 0; opacity: 0.2;'>", unsafe_allow_html=True)
        st.markdown("### 📝 Transcription Results")
        
        # Audio info
        info = st.session_state['audio_info']
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Duration", info['duration_str'])
        with col2:
            st.metric("Processing Time", f"{st.session_state['processing_time']:.1f}s")
        with col3:
            st.metric("Speed", f"{info['duration'] / st.session_state['processing_time']:.1f}x")
        
        # Transcription text with better styling
        st.markdown("**Transcribed Text**")
        transcribed_text = st.text_area(
            "Transcribed Text",
            value=st.session_state['transcription'],
            height=300,
            help="You can edit the text before copying or downloading",
            label_visibility="collapsed",
            key="transcription_output"
        )
        
        # Action buttons in columns
        action_col1, action_col2 = st.columns(2)
        
        with action_col1:
            # Copy to clipboard using st.code
            with st.expander("📋 Copy to Clipboard"):
                st.code(transcribed_text, language=None)
                st.caption("Click the copy button above")
        
        with action_col2:
            # Download button
            st.download_button(
                label="💾 Download as .txt",
                data=transcribed_text,
                file_name=f"transcription_{int(time.time())}.txt",
                mime="text/plain",
                use_container_width=True
            )
    
    # Footer with better styling
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; padding: 2rem 0; opacity: 0.6; font-size: 0.9rem;'>
        🔒 All processing happens locally on your machine<br>
        No data is sent to external servers
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()