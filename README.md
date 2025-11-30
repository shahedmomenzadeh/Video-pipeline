# Video Processing Pipeline (Modular)

This project is a comprehensive pipeline designed to create high-quality text datasets from YouTube videos (specifically targeted at medical/surgical content, but applicable generally).

The pipeline automates the process of downloading videos, extracting audio, transcribing speech using OpenAI Whisper, refining the text using the Cerebras LLM, and generating a final summary dataset.

## 🚀 Features

The pipeline runs in 5 sequential steps:

1. **Ingestion**: Downloads videos and extracts audio (16kHz WAV) using yt-dlp and FFmpeg.
2. **Hygiene**: Automatically deletes videos that exceed a specific duration threshold (to avoid processing overly long files).
3. **Transcription**: Generates timestamped transcripts using OpenAI's Whisper model.
4. **Refinement**: Uses the Cerebras LLM to correct grammar and medical terminology while preserving timestamps.
5. **Reporting**: Merges metadata and processing logs into a final dataset_info.csv summary file.

## 🛠️ Prerequisites

Before running the project, ensure you have the following installed:

- **Python 3.8+**
- **FFmpeg**: This is critical for audio extraction.
  - Ubuntu/Debian: ```sudo apt update && sudo apt install ffmpeg ```
  - MacOS: ```brew install ffmpeg ```
  - Windows: Download FFmpeg and add it to your System PATH.
- **Cerebras API Key**: You need an API key to run the refinement step.

## 📥 Installation

### Clone the Repository

Use the specific branch `modular` for this pipeline:

```bash
git clone --branch modular --single-branch https://github.com/shahedmomenzadeh/Video-pipeline.git
cd Video-pipeline
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## ⚙️ Configuration

### 1. The config.yaml File

All settings are managed in `config.yaml`. You do not need to modify the Python code directly.

```yaml
# Directory Configuration
directories:
  videos: "./videos"              # Where raw videos are downloaded
  audio: "./audio"                # Where extracted WAV files are stored
  transcripts: "./transcripts"    # Raw Whisper JSON outputs
  refined_transcripts: "./refined_transcripts" # Final LLM-cleaned text

# Download Settings
download:
  max_duration_seconds: 1500      # Videos longer than this (25 mins) will be deleted

# Transcription Settings
whisper:
  model_size: "large"             # Options: tiny, base, small, medium, large
  device: "cuda"                  # Use 'cuda' for GPU, 'cpu' for CPU
```

### 2. Adding Video Links

Open the `videos_link.txt` file and add your YouTube URLs (videos or playlists), one per line:

```
https://youtu.be/example1
https://youtu.be/example2
```

### 3. YouTube Cookies (Crucial for Age-Restricted Content)

To avoid download errors with restricted videos, you must provide a cookies file.

1. Install the Get cookies.txt LOCALLY Chrome extension: https://chromewebstore.google.com/detail/get-cookiestxt-locally/cclelndahbckbenkjhflpdbgdldlbecc
2. Go to YouTube.com and log in.
3. Click the extension icon and select "Export".
4. Rename the downloaded file to `www.youtube.com_cookies.txt`.
5. Place this file in the root directory of the project.

## 🖥️ How to Run Locally

### Set up your Environment Variables

Create a file named `.env` in the root directory and add your key:

```bash
CEREBRAS_API_KEY=your_actual_api_key_here
```

### Run the Pipeline

```bash
python main.py
```

The script will automatically detect the key in the `.env` file.

## ☁️ How to Run in Google Colab

The script is optimized for Colab. You do not need a `.env` file there.

1. Clone the repository inside a Colab cell.
2. Add your API Key to Colab Secrets:
   - Click the Key icon (Secrets) on the left sidebar.
   - Name: `CEREBRAS_API_KEY`
   - Value: `your_actual_key`
   - Toggle "Notebook access" to On.
3. Run the Pipeline using this code block:

```python
from google.colab import userdata
import os

# 1. Clone the repo (if not already done)
!git clone --branch modular --single-branch https://github.com/shahedmomenzadeh/Video-pipeline.git
%cd Video-pipeline

# 2. Install requirements
!pip install -r requirements.txt

# 3. Get the API Key securely
try:
    api_key = userdata.get('CEREBRAS_API_KEY')
except Exception:
    api_key = ""
    print("Warning: CEREBRAS_API_KEY not found in Secrets.")

# 4. Run the main script, passing the key as an argument
!python main.py "$api_key"
```

## 📂 Project Structure

```
Project_Root/
├── config.yaml              # Central configuration settings
├── main.py                  # Entry point script
├── videos_link.txt          # Input: List of YouTube URLs
├── www.youtube.com_cookies.txt  # Input: Cookies for auth (User provided)
├── dataset_info.csv         # Output: Final Summary CSV
│
├── modules/                 # Logic Modules
│   ├── downloader.py        # yt-dlp & FFmpeg logic
│   ├── cleaner.py           # Duration filtering
│   ├── transcriber.py       # Whisper logic
│   ├── refiner.py           # LLM logic
│   └── summarizer.py        # CSV merging logic
│
├── videos/                  # Stores metadata CSV
├── audio/                   # Stores audio files
├── transcripts/             # Stores raw JSON
└── refined_transcripts/     # Stores final TXT files & logs
```

## ⚠️ Common Issues

- **FFmpeg Error**: If you see FFmpeg is not installed, you cannot extract audio. Install it via your OS package manager.
- **Sign-in Required**: If yt-dlp fails with "Sign in to confirm you're not a bot", ensure your `www.youtube.com_cookies.txt` is fresh and placed in the root folder.
- **GPU Memory**: If running locally with a small GPU, change `model_size` in `config.yaml` from `large` to `medium` or `small`.
