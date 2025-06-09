# AudioInk 🎧

**Professional local audio transcription with Whisper-powered language detection**

Transform audio/video files and YouTube URLs into accurate text with real-time processing and intelligent language detection.

## ⚡ Quick Start

```bash
# 1. Clone and install
git clone https://github.com/fabianhtml/AudioInk.git
cd AudioInk
pip install -r requirements.txt

# 2. Install FFmpeg
brew install ffmpeg  # macOS
# sudo apt install ffmpeg  # Ubuntu/Debian

# 3. Run AudioInk
./start_audioink.sh  # Background mode (recommended)
# streamlit run audioink.py  # Normal mode
```

**Open**: `http://localhost:8501`

## 🎯 Key Features

### **🎧 Whisper-Powered Language Detection**
- **95% accuracy** language identification from audio analysis
- Supports **99+ languages** automatically
- Analyzes first 5 seconds for instant detection

### **⚡ Dual Input Modes**
- **YouTube URLs**: Instant transcription + Whisper fallback
- **File Upload**: Real-time processing with live preview
- **Formats**: MP3, WAV, M4A, FLAC, OGG, MP4, AVI, MOV

### **🚀 Smart Processing** 
- Automatic chunking for large files (>2 min)
- Settings lock during transcription
- Background processing support
- Professional dark theme UI

## 🔧 Technical Details

### **Whisper Models Available**
| Model | Parameters | Quality | Speed | VRAM Required |
|-------|------------|---------|-------|---------------|
| **large-v3-turbo** ⭐ | 809M | Excellent | 8x faster | ~6GB |
| tiny | 39M | Basic | 10x faster | ~1GB |
| base | 74M | Good | 7x faster | ~1GB |
| small | 244M | Better | 4x faster | ~2GB |
| medium | 769M | High | 2x faster | ~5GB |
| large | 1550M | Best | 1x baseline | ~10GB |

### **System Requirements**
- **Python**: 3.8+
- **VRAM/RAM**: 1-10GB (model dependent)
- **Storage**: 200MB-6GB for models
- **FFmpeg**: Required for audio processing

## 🏗️ Architecture

**Modular Design** (6 specialized modules):
- `audioink.py` - Main orchestration
- `youtube_handler.py` - YouTube processing + Whisper detection  
- `audio_processing.py` - Whisper transcription engine
- `ui_components.py` - Streamlit interface
- `constants.py` - Configuration & settings
- `utils.py` - Helper functions

## 📋 Usage Commands

```bash
# Start in background (recommended)
./start_audioink.sh

# Start normally  
streamlit run audioink.py

# Stop background process
pkill -f "streamlit run audioink.py"
```

## 🛡️ Privacy & Security

- **100% Local Processing**: No data sent to external servers
- **Temporary Files**: YouTube audio auto-deleted after transcription
- **Offline Capable**: Works without internet (after initial model download)

---

## 📄 License

MIT License - Open source and free to use.

**AudioInk** - Transcription tool powered by OpenAI Whisper  
Made with ❤️ for accurate, local audio transcription
