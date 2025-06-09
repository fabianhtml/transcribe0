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
| Model | Quality | Speed | Size |
|-------|---------|-------|------|
| **large-v3-turbo** ⭐ | Excellent | Fast | ~1.5GB |
| tiny | Basic | Ultra Fast | ~39MB |
| base | Good | Fast | ~74MB |
| small | Better | Medium | ~244MB |
| medium | High | Slow | ~769MB |
| large | Best | Slowest | ~1.5GB |

### **System Requirements**
- **Python**: 3.8+
- **RAM**: 2-8GB (model dependent)
- **Storage**: 1-4GB for models
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
