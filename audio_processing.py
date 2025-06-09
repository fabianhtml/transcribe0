"""
AudioInk - Audio Processing Module
Handles Whisper transcription and audio file processing
"""

import streamlit as st
import whisper
import tempfile
import os
from pathlib import Path
from pydub import AudioSegment
from pydub.utils import make_chunks

from constants import AudioConstants, UIConstants

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