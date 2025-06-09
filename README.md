# AudioInk - Local Audio Transcription Tool

A professional local audio-to-text transcription tool with instant YouTube transcription and high-quality Whisper processing.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/fabianhtml/AudioInk.git
cd AudioInk
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
streamlit run audioink.py
```

The app will open in your browser at `http://localhost:8501`

### How It Works

1. **YouTube URLs**: Paste any YouTube URL → Get instant transcription (if available) or high-quality Whisper processing
2. **File Upload**: Drop audio/video files → Real-time Whisper transcription with live progress
3. **Smart Processing**: Automatic chunking for large files, optimal model selection, settings lock during processing

## Key Features

- **⚡ Instant YouTube Transcription**: Get immediate results from YouTube's existing transcripts
- **🎯 High-Quality Whisper**: large-v3-turbo model for maximum accuracy when needed  
- **📱 Smart Interface**: Settings lock during processing, one-click navigation, elegant dark theme
- **🔄 Real-time Display**: Watch transcription appear live with chunked processing
- **💾 Multiple Export**: Copy to clipboard, download as .txt with intelligent naming
- **🌍 Multi-language**: Auto-detect or choose from 6 languages
- **📁 Multi-format**: MP3, WAV, M4A, FLAC, OGG, MP4, AVI, MOV support

## Models

OpenAI Whisper models (default: **large-v3-turbo**):

- **large-v3-turbo**: Excellent quality, faster than large ⭐ *Default*
- **tiny**: Ultra fast, lower quality (~39 MB)
- **base**: Balanced speed and quality (~74 MB)
- **small**: Good quality, moderate speed (~244 MB)
- **medium**: High quality, slower (~769 MB)
- **large**: Best quality, slowest (~1550 MB)

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
