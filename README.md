# Transcribe0 - Local Audio Transcription Tool

A dark-themed local audio-to-text transcription tool using OpenAI Whisper and Streamlit. Features real-time transcription display, YouTube integration, and intelligent chunking for large files.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/transcribe0.git
cd transcribe0
```

2. Install Python dependencies:

```bash
pip install -r requirements.txt
```

3. Install FFmpeg (required for audio processing):

- **macOS**: `brew install ffmpeg`
- **Ubuntu/Debian**: `sudo apt update && sudo apt install ffmpeg`
- **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/download.html)

Note: For Python 3.13+, you may need to install `audioop-lts`:
```bash
pip install audioop-lts
```

4. For YouTube support, yt-dlp is included in requirements.txt

## Usage

Run the application:
```bash
streamlit run transcribe0.py
```

The app will open in your browser at `http://localhost:8501`

### Input Methods

1. **File Upload**: Drag and drop or select audio/video files
2. **YouTube URL**: Paste any YouTube video URL to transcribe its audio

The app automatically downloads the audio from YouTube videos and processes them locally.

## Features

### Core Functionality

- **100% Local Processing**: All transcription happens on your machine - no data sent to external servers
- **Real-time Live Display**: Watch transcription appear as it's generated chunk by chunk
- **Smart Chunking**: Automatic processing for large files (>2 minutes) with live progress updates
- **Multiple Input Sources**: Upload files or paste YouTube URLs directly

### Interface & Usability

- **Professional Dark Theme**: Eye-friendly interface with subtle color scheme
- **Locked Settings During Processing**: Prevents accidental changes during transcription
- **Progress Indicators**: Clear feedback on download and transcription progress
- **Intelligent File Naming**: Downloaded transcripts use original video/file names

### Technical Capabilities

- **6 Whisper Models**: From ultra-fast 'tiny' to high-quality 'large-v3-turbo' (default)
- **Multi-format Support**: MP3, WAV, M4A, FLAC, OGG, MP4, AVI, MOV
- **6 Language Options**: Auto-detect or manual selection (EN, ES, FR, DE, IT, PT)
- **YouTube Integration**: Direct audio extraction from YouTube videos (temporary download, auto-cleanup)

### Export & Sharing

- **Multiple Export Options**: Copy to clipboard or download as .txt file
- **Word Count Metrics**: Real-time statistics on transcribed content
- **Clean Filename Generation**: Automatic sanitization of special characters

## Models
OpenAI Whisper models:

- **tiny**: Ultra fast, lower quality
- **base**: Balanced speed and quality
- **small**: Good quality, moderate speed
- **medium**: High quality, slower
- **large**: Best quality, slowest
- **large-v3-turbo**: Excellent quality, faster than large

First run will download the selected model automatically.

## System Requirements

- Python 3.8+
- 2-8 GB RAM (depending on model size)
- FFmpeg installed
- Internet connection (only for first model download)

## User Experience

### Professional Interface

- **Dark Theme**: Subtle blue-gray color scheme (#6B7280) with excellent contrast
- **Tabbed Input**: Clean separation between file upload and YouTube URL input
- **Real-time Feedback**: Immediate validation of YouTube URLs with video metadata display

### Smart Workflow

- **Locked Controls**: Settings automatically lock during processing to prevent interference
- **Progress Tracking**: Live transcription display shows text appearing in real-time
- **Automatic Cleanup**: YouTube audio files are downloaded temporarily and deleted after transcription

### Quality Optimizations

- **Default Model**: large-v3-turbo (OpenAI Whisper)
selected by default for optimal speed/quality balance
- **Intelligent Chunking**: Files over 2 minutes are processed in 1-minute segments
- **Error Handling**: Graceful recovery from network issues or unsupported formats

## License

MIT License - feel free to use and modify as needed.
