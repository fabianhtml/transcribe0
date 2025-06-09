"""
AudioInk - YouTube Handler Module
Handles YouTube URL processing, subtitle extraction, and audio downloading
"""

import tempfile
import os
import yt_dlp
import whisper
from pydub import AudioSegment

from constants import YOUTUBE_REGEX, AudioConstants
from utils import clean_subtitle_text

def detect_language_with_whisper(url, duration_seconds=5):
    """Detect language using Whisper by analyzing first few seconds of audio"""
    try:
        # Download just a small sample of audio for language detection
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, '%(title)s.%(ext)s')
            
            ydl_opts = {
                'format': 'bestaudio/best',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '96',  # Lower quality for faster download
                }],
                'outtmpl': output_path,
                'quiet': True,
                'no_warnings': True,
                'extract_flat': False,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.extract_info(url, download=True)
                
                # Find the downloaded audio file
                audio_file = None
                for file in os.listdir(temp_dir):
                    if file.endswith('.mp3'):
                        audio_file = os.path.join(temp_dir, file)
                        break
                
                if not audio_file:
                    return 'unknown'
                
                # Extract just the first N seconds
                audio = AudioSegment.from_mp3(audio_file)
                sample_audio = audio[:duration_seconds * 1000]  # Convert to milliseconds
                
                # Save the sample
                sample_path = os.path.join(temp_dir, 'sample.wav')
                sample_audio.export(sample_path, format='wav')
                
                # Use Whisper's language detection (faster model for detection)
                model = whisper.load_model('base')  # Smaller model for quick detection
                
                # Detect language without full transcription
                audio_data = whisper.load_audio(sample_path)
                audio_data = whisper.pad_or_trim(audio_data)
                mel = whisper.log_mel_spectrogram(audio_data).to(model.device)
                
                # Detect language
                _, probs = model.detect_language(mel)
                detected_language = max(probs, key=probs.get)
                confidence = probs[detected_language]
                
                print(f"🎧 Whisper language detection: {detected_language} (confidence: {confidence:.2f})")
                
                # Only trust high confidence detections
                if confidence > 0.5:
                    return detected_language
                else:
                    print(f"⚠️ Low confidence ({confidence:.2f}), falling back to subtitle analysis")
                    return 'unknown'
                    
    except Exception as e:
        print(f"⚠️ Whisper detection failed: {str(e)}, falling back to subtitle analysis")
        return 'unknown'

def detect_language_from_content(text):
    """Detect language by analyzing subtitle content"""
    if not text or len(text) < 50:
        return 'unknown'
    
    text_lower = text.lower()
    
    # English indicators
    english_words = ['the', 'and', 'that', 'have', 'for', 'not', 'with', 'you', 'this', 'but', 'his', 'from', 'they', 'she', 'her', 'been', 'than', 'its', 'who', 'did']
    english_score = sum(1 for word in english_words if f' {word} ' in text_lower)
    
    # Spanish indicators  
    spanish_words = ['que', 'una', 'con', 'para', 'por', 'son', 'como', 'este', 'esta', 'pero', 'sus', 'desde', 'ellos', 'ella', 'sido', 'más', 'quien', 'hizo']
    spanish_score = sum(1 for word in spanish_words if f' {word} ' in text_lower)
    
    # French indicators
    french_words = ['que', 'une', 'avec', 'pour', 'par', 'sont', 'comme', 'cette', 'mais', 'ses', 'depuis', 'ils', 'elle', 'été', 'plus', 'qui', 'fait']
    french_score = sum(1 for word in french_words if f' {word} ' in text_lower)
    
    # Determine language based on scores
    scores = {
        'en': english_score,
        'es': spanish_score, 
        'fr': french_score
    }
    
    if max(scores.values()) == 0:
        return 'unknown'
        
    detected_lang = max(scores, key=scores.get)
    
    # Require a minimum confidence (at least 2 word matches)
    if scores[detected_lang] >= 2:
        return detected_lang
    
    return 'unknown'

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

def get_youtube_subtitles(url, language='auto', use_whisper_detection=True):
    """Get existing YouTube subtitles if available with intelligent language detection"""
    
    # First, try Whisper-based language detection for most accurate results
    whisper_detected_lang = 'unknown'
    if language == 'auto' and use_whisper_detection:
        print("🎧 Using Whisper for language detection...")
        whisper_detected_lang = detect_language_with_whisper(url)
    
    # Then get video info to detect the original language
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
        
        # Create smart priority list based on Whisper detection
        if whisper_detected_lang != 'unknown' and whisper_detected_lang in available_langs:
            # Prioritize Whisper-detected language (most accurate)
            subtitle_langs = [whisper_detected_lang]
            print(f"🎯 Using Whisper-detected language: {whisper_detected_lang}")
            # Add other common languages as fallback
            for lang in ['es', 'en', 'fr', 'de', 'it', 'pt']:
                if lang != whisper_detected_lang and lang in available_langs:
                    subtitle_langs.append(lang)
        elif video_language and video_language in available_langs:
            # Fallback to YouTube's detected video language
            subtitle_langs = [video_language]
            print(f"🎯 Using YouTube-detected language: {video_language}")
            # Add other common languages as fallback
            for lang in ['es', 'en', 'fr', 'de', 'it', 'pt']:
                if lang != video_language and lang in available_langs:
                    subtitle_langs.append(lang)
        else:
            # Final fallback: intelligent language priority based on available content
            # Check if English is heavily represented (indicates English content)
            english_indicators = sum(1 for lang in available_langs if lang.startswith('en'))
            spanish_indicators = sum(1 for lang in available_langs if lang.startswith('es'))
            
            if english_indicators >= spanish_indicators:
                # Prioritize English when English content is detected
                priority_order = ['en', 'es', 'fr', 'de', 'it', 'pt']
            else:
                # Prioritize Spanish when Spanish content is more prevalent
                priority_order = ['es', 'en', 'fr', 'de', 'it', 'pt']
                
            subtitle_langs = [lang for lang in priority_order if lang in available_langs]
            print(f"🎯 Using availability-based priority: {priority_order[0] if priority_order else 'unknown'}")
            
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
                        
                        # Use Whisper detection as the ultimate authority if available
                        if whisper_detected_lang != 'unknown':
                            # Whisper detection overrides everything else
                            original_detected = detected_language
                            detected_language = whisper_detected_lang
                            
                            if original_detected != whisper_detected_lang:
                                print(f"🎧 Language corrected by Whisper: {original_detected} → {whisper_detected_lang}")
                        else:
                            # Fallback to content analysis if Whisper failed
                            content_language = detect_language_from_content(cleaned_text)
                            if content_language != 'unknown':
                                # Content analysis overrides filename-based detection
                                original_detected = detected_language
                                detected_language = content_language
                                
                                # Debug info (visible in console/logs)
                                if original_detected != content_language:
                                    print(f"🔍 Language corrected by content analysis: {original_detected} → {content_language}")
                            else:
                                print(f"🔍 Using filename-based detection: {detected_language}")
                            
                        print(f"🎯 Final language: {detected_language}")
                        
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