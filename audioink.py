#!/usr/bin/env python3
"""
AudioInk - Local Audio Transcription Tool
Uses OpenAI Whisper for offline audio-to-text conversion
"""

import streamlit as st
import tempfile
import os
import time

# Import modular components
from ui_components import (
    render_header, initialize_session_state, render_input_section,
    render_settings_section, determine_audio_source, render_results_section,
    render_footer
)
from audio_processing import transcribe_audio, transcribe_audio_file
from youtube_handler import download_youtube_audio

# Page configuration
st.set_page_config(
    page_title="AudioInk",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def handle_transcription_process(audio_source, source_name, model_name, language, transcribing):
    """Handle the transcription workflow"""
    if audio_source is None:
        return
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Check if YouTube subtitles are already shown
    youtube_subtitles_already_shown = st.session_state.get('youtube_subtitles_shown', False)
    
    if not transcribing and not youtube_subtitles_already_shown:
        if st.button("Start Transcription", type="primary", use_container_width=True):
            # Lock settings and start transcription
            st.session_state['transcribing'] = True
            st.session_state['locked_model'] = model_name
            st.session_state['locked_language'] = language
            st.session_state['locked_use_youtube'] = False  # Always use Whisper when manually triggered
            st.rerun()
    elif youtube_subtitles_already_shown:
        # The regenerate button is now handled in the UI components
        pass
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
    
    # Display results only if we have an active audio source
    render_results_section(uploaded_file, youtube_url)
    
    # Footer
    render_footer()

if __name__ == "__main__":
    main()