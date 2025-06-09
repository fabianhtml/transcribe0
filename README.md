# Transcribe0 - Local Audio Transcription Tool

A simple, local audio-to-text transcription tool using OpenAI Whisper and Streamlit.

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Install FFmpeg (required for audio processing):
- **macOS**: `brew install ffmpeg`
- **Ubuntu/Debian**: `sudo apt update && sudo apt install ffmpeg`
- **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/download.html)

## Usage

Run the application:
```bash
streamlit run transcribe0.py
```

The app will open in your browser at `http://localhost:8501`

## Features

- **Local Processing**: All transcription happens on your machine
- **Multiple Formats**: Supports MP3, WAV, M4A, FLAC, OGG, MP4, AVI, MOV
- **Model Selection**: Choose from tiny to large models based on your needs
- **Language Support**: Auto-detect or manual selection for 6 languages
- **Large File Handling**: Automatic chunking for files over 5 minutes
- **Export Options**: Copy to clipboard or download as .txt file

## Models

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