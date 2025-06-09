"""
AudioInk - UI Components Module
Streamlit interface components and rendering functions
"""

import streamlit as st
import time

from constants import MODELS, LANGUAGES, ALL_FORMATS, UIConstants
from youtube_handler import is_youtube_url, get_youtube_info, get_youtube_subtitles
from utils import has_youtube_transcription, has_file_transcription, clear_youtube_data, clear_file_data

@st.cache_data
def render_video_info_card(title, uploader, duration):
    """Render video information card with caching"""
    return UIConstants.VIDEO_INFO_STYLE.format(title, uploader, duration // 60, duration % 60)

@st.cache_data
def render_subtitle_info_card(detected_lang, word_count):
    """Render subtitle availability card with caching"""
    return UIConstants.SUBTITLE_AVAILABLE_STYLE.format(
        detected_lang.upper(), word_count
    )

def render_header():
    """Render application header"""
    st.markdown("""
    <h1 style='text-align: center; margin-bottom: 0;'>AudioInk</h1>
    <p style='text-align: center; color: #666; margin-top: 0;'>Local audio-to-text transcription with Whisper from OpenAI for free</p>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'transcribing' not in st.session_state:
        st.session_state['transcribing'] = False
    return st.session_state.get('transcribing', False)

def render_input_section(transcribing):
    """Handle file upload and YouTube URL input with proper tab isolation"""
    
    # Initialize active tab state
    if 'active_input_tab' not in st.session_state:
        st.session_state.active_input_tab = 'file'
    
    # Custom tab selector with proper state management
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📁 Upload File", 
                    type="primary" if st.session_state.active_input_tab == 'file' else "secondary",
                    use_container_width=True,
                    disabled=transcribing):
            st.session_state.active_input_tab = 'file'
            st.rerun()
    
    with col2:
        if st.button("🔗 YouTube URL", 
                    type="primary" if st.session_state.active_input_tab == 'youtube' else "secondary",
                    use_container_width=True,
                    disabled=transcribing):
            st.session_state.active_input_tab = 'youtube'
            st.rerun()
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    uploaded_file = None
    youtube_url = None
    
    # Render content based on active tab
    if st.session_state.active_input_tab == 'file':
        uploaded_file = st.file_uploader(
            "Drag and drop or click to browse",
            type=ALL_FORMATS,
            help=f"Supported formats: {', '.join(ALL_FORMATS)}",
            label_visibility="collapsed",
            disabled=transcribing,
            key="file_upload_widget"
        )
        
        # Clear YouTube data only when file is actually uploaded (conflict resolution)
        if uploaded_file is not None and has_youtube_transcription():
            clear_youtube_data()
            
    elif st.session_state.active_input_tab == 'youtube':
        youtube_url = st.text_input(
            "Enter YouTube URL",
            placeholder="https://www.youtube.com/watch?v=...",
            help="Paste a YouTube video URL to transcribe its audio",
            label_visibility="collapsed",
            disabled=transcribing,
            key="youtube_url_field"
        )
        st.caption("ℹ️ Audio is downloaded temporarily and deleted after transcription")
        
        if youtube_url and is_youtube_url(youtube_url):
            # Clear file data only when YouTube URL is entered and there's file transcription (conflict resolution)
            if 'file_upload_widget' in st.session_state and st.session_state.file_upload_widget and has_file_transcription():
                clear_file_data()
                
            st.markdown('<p style="color: #6B7280;">✅ Valid YouTube URL</p>', unsafe_allow_html=True)
            
            # Cache video info in session state by URL
            cache_key = f"video_info_{hash(youtube_url)}"
            if cache_key not in st.session_state:
                try:
                    with st.spinner('🔍 Getting video info and detecting language...'):
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
                    
                    st.markdown(
                        render_subtitle_info_card(detected_lang, word_count),
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
                        
                        # Show regenerate button for YouTube transcriptions
                        st.markdown("<br>", unsafe_allow_html=True)
                        if st.button("🔄 Regenerate with Whisper", type="secondary", use_container_width=True):
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

def render_results_section(uploaded_file=None, youtube_url=None):
    """Display transcription results and export options"""
    if 'transcription' not in st.session_state:
        return
    
    # Get current active tab and transcription source
    active_tab = st.session_state.get('active_input_tab', 'file')
    transcription_source = st.session_state.get('transcription_source', '')
    
    # Check if current transcription matches active tab
    is_youtube_transcription = 'YouTube' in transcription_source
    is_file_transcription = 'Whisper' in transcription_source
    
    # Only show results if transcription source matches active tab
    if active_tab == 'file' and is_youtube_transcription:
        return  # Don't show YouTube results in file tab
    if active_tab == 'youtube' and is_file_transcription:
        return  # Don't show file results in YouTube tab
        
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