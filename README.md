# Video Processing Pipeline (Modular)

This project is a comprehensive pipeline designed to create high-quality text and Vision-Language Model (VLM) datasets from YouTube videos (specifically targeted at medical/surgical content, but applicable generally).

The pipeline automates the process of downloading videos, extracting audio, transcribing speech using OpenAI Whisper, refining the text using the Cerebras LLM, generating a structured VLM dataset using Google Gemini models, detecting surgical adverse events, and optionally splitting videos into clips.

## 🚀 Features

The pipeline runs in 8 sequential steps:

1. **Ingestion**: Downloads videos and extracts audio (16kHz WAV) using yt-dlp and FFmpeg.
2. **Hygiene**: Automatically deletes videos that exceed a specific duration threshold (to avoid processing overly long files).
3. **Transcription**: Generates timestamped transcripts using OpenAI's Whisper model.
4. **Refinement**: Uses the Cerebras LLM to correct grammar and medical terminology while preserving timestamps.
5. **Reporting**: Merges metadata and processing logs into a final `dataset_info.csv` summary file.
6. **VLM Dataset Generation**: Uses a two-stage Google Gemini pipeline (Gatekeeper & Analyst) to generate a structured JSONL dataset for VLM fine-tuning, including visual descriptions, surgical steps, and instrument identification.
7. **Adverse Event Detection**: Analyzes VLM annotations to identify intraoperative complications (e.g., zonular dialysis, posterior capsule rupture, iris prolapse) using Google Gemini's safety analysis model.
8. **Video Splitting** (Optional): Extracts clips from original videos based on detected adverse events or VLM annotations.

## 🛠️ Prerequisites

Before running the project, ensure you have the following installed:

- **Python 3.8+**
- **FFmpeg**: This is critical for audio extraction.
  - Ubuntu/Debian: `sudo apt update && sudo apt install ffmpeg`
  - MacOS: `brew install ffmpeg`
  - Windows: Download FFmpeg and add it to your System PATH.
- **API Keys**:
  - **Cerebras API Key**: For the text refinement step.
  - **Google Gemini API Key**: For the VLM dataset generation and adverse event detection steps.

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

All settings are managed in `config.yaml`.

```yaml
# Directory Configuration
directories:
  videos: "./videos"              # Where raw videos are downloaded
  audio: "./audio"                # Where extracted WAV files are stored
  transcripts: "./transcripts"    # Raw Whisper JSON outputs
  refined_transcripts: "./refined_transcripts" # Final LLM-cleaned text
  vlm_dataset: "./vlm_dataset"    # Where JSONL files are stored
  adverse_events: "./adverse_events" # Where adverse event analysis results are stored
  clips: "./clips"                # Where extracted video clips are stored

# Download Settings
download:
  max_duration_seconds: 1500      # Videos longer than this (25 mins) will be deleted

# Transcription Settings
whisper:
  model_size: "large"             # Options: tiny, base, small, medium, large
  device: "cuda"                  # Use 'cuda' for GPU, 'cpu' for CPU

# Cerebras Settings
cerebras:
  model: "claude-3.5-sonnet"      # Model name for text refinement
  api_call_delay_seconds: 2       # Delay between API calls
  max_files_per_run: 10           # Maximum files to process per run

# File Settings
files:
  metadata_csv: "metadata.csv"    # Metadata file for videos
  cookies: "www.youtube.com_cookies.txt" # YouTube cookies
  refinement_log: "refine_log.csv" # Log for refinement step
  video_links: "videos_link.txt"  # File containing YouTube URLs
  dataset_summary: "dataset_info.csv" # Final summary file

# VLM Settings
vlm:
  gatekeeper_model: "gemini-2.0-flash" # Fast model for quality checks
  generator_model: "gemini-2.0-flash"  # Powerful model for video analysis
  aggregate_file: "vlm_dataset_all.jsonl"
  log_file: "process_log.csv"

# Adverse Event Detection Settings
adverse_event:
  model: "gemini-2.0-flash"       # Gemini model for safety analysis
  aggregate_file: "adverse_events_all.jsonl" # Aggregate file for all detected events
  log_file: "adverse_event_log.csv" # Log file tracking analysis results

# Video Splitter Settings
splitter:
  log_file: "splitter_log.csv"    # Log file for video splitting
```

### 2. Adding Video Links

Open the `videos_link.txt` file and add your YouTube URLs (videos or playlists), one per line:

```
https://youtu.be/example1
https://youtu.be/example2
```

### 3. YouTube Cookies (Crucial for Age-Restricted Content)

To avoid download errors with restricted videos, you must provide a cookies file.

1. Install the "Get cookies.txt" Chrome extension.
2. Go to YouTube.com and log in.
3. Click the extension icon and select "Export".
4. Rename the downloaded file to `www.youtube.com_cookies.txt`.
5. Place this file in the root directory of the project.

## 🖥️ How to Run Locally

### Set up your Environment Variables

Create a file named `.env` in the root directory and add your keys:

```bash
CEREBRAS_API_KEY=your_cerebras_key
GEMINI_API_KEY=your_gemini_key
```

### Run the Pipeline

You can run the entire pipeline or specific steps using the `--step` argument.

**Run Everything (Steps 1-8):**

```bash
python main.py
```

**Run Only Specific Steps:**

```bash
python main.py --step download    # Step 1: Download videos
python main.py --step clean       # Step 2: Clean videos by duration
python main.py --step transcribe  # Step 3: Transcribe audio
python main.py --step refine      # Step 4: Refine transcripts
python main.py --step summarize   # Step 5: Create dataset summary
python main.py --step vlm         # Step 6: Generate VLM dataset
python main.py --step adverse_event  # Step 7: Detect adverse events
python main.py --step split       # Step 8: Split videos into clips
```

**Available Steps:** `download`, `clean`, `transcribe`, `refine`, `summarize`, `vlm`, `adverse_event`, `split`, `all`.

## ☁️ How to Run in Google Colab

The script is optimized for Colab. You do not need a `.env` file there.

1. Clone the repository inside a Colab cell.
2. Add your API Keys to Colab Secrets:
   - Click the Key icon (Secrets) on the left sidebar.
   - Name: `CEREBRAS_API_KEY`, Value: `your_key`
   - Name: `GEMINI_API_KEY`, Value: `your_key`
   - Toggle "Notebook access" to **On** for both.
3. Run the Pipeline using this code block:

```python
from google.colab import userdata
import os

# 1. Clone the repo (if not already done)
!git clone --branch modular --single-branch https://github.com/shahedmomenzadeh/Video-pipeline.git
%cd Video-pipeline

# 2. Install requirements
!pip install -r requirements.txt

# 3. Get the API Keys securely
os.environ['CEREBRAS_API_KEY'] = userdata.get('CEREBRAS_API_KEY')
os.environ['GEMINI_API_KEY'] = userdata.get('GEMINI_API_KEY')

# 4. Run the main script
# To run full pipeline:
!python main.py

# To run ONLY the adverse event detection step:
# !python main.py --step adverse_event
```

## 📂 Project Structure

```
Project_Root/
├── config.yaml              # Central configuration settings
├── main.py                  # Entry point script (CLI enabled)
├── videos_link.txt          # Input: List of YouTube URLs
├── www.youtube.com_cookies.txt  # Input: Cookies for auth (User provided)
├── dataset_info.csv         # Output: Final Summary CSV
│
├── modules/                 # Logic Modules
│   ├── downloader.py        # yt-dlp & FFmpeg logic
│   ├── cleaner.py           # Duration filtering
│   ├── transcriber.py       # Whisper logic
│   ├── refiner.py           # LLM logic (Cerebras)
│   ├── summarizer.py        # CSV merging logic
│   ├── vlm_generator.py     # VLM Dataset logic (Gemini)
│   ├── adverse_event_detector.py # Surgical safety analysis (Gemini)
│   └── splitter.py          # Video clipping logic
│
├── videos/                  # Stores raw video & metadata
├── audio/                   # Stores audio files
├── transcripts/             # Stores raw JSON
├── refined_transcripts/     # Stores refined TXT files & logs
├── vlm_dataset/             # Stores final JSONL datasets & VLM logs
├── adverse_events/          # Stores adverse event analysis results & logs
└── clip_dataset/                   # Stores extracted video clips
```

## 🔍 Adverse Event Detection Details

### Overview

The Adverse Event Detector (Step 7) analyzes VLM-annotated surgical videos to identify intraoperative complications. It uses Google Gemini to process visual descriptions and detect safety-critical events.

### How It Works

1. **Input**: Reads individual JSONL files from the VLM Dataset folder (containing visual descriptions of surgical steps)
2. **Analysis**: Sends surgical timeline (visual descriptions) to Google Gemini with a specialized safety analysis prompt
3. **Detection**: Identifies complications based on visual descriptions and surgical management actions (e.g., anterior vitrectomy implies posterior capsule rupture)
4. **Output**: Saves detected events to individual and aggregate JSONL files, logs all processing status

### Detected Complications

The module identifies the following surgical adverse events:

**Intraoperative Complications:**
- Iris Prolapse
- Zonular Dialysis
- IFIS (Intraoperative Floppy Iris Syndrome)
- Phaco Wound Burn
- Posterior Capsule Rupture (PCR)
- Vitreous Loss
- Nucleus Drop
- IOL Drop

**Retinal/Posterior Segment Complications:**
- Peripheral Retinal Tear
- Retinal Hemorrhage

### Output Files

- **Individual JSONL Files**: One file per video with detected adverse events (e.g., `video_001.jsonl`)
  - Contains metadata from VLM dataset plus detected `adverse_events` array
- **Aggregate JSONL**: `adverse_events_all.jsonl` containing all videos with detected events
- **Log CSV**: `adverse_event_log.csv` with columns:
  - `video_id`: Video identifier
  - `status`: Processing status (`DETECTED`, `NO_EVENT`, or `ERROR`)
  - `event_count`: Number of adverse events found
  - `timestamp`: Processing completion time

### Resume Logic

The detector includes automatic resume capability. If processing is interrupted:
- It checks the log CSV to identify already-processed videos
- Skips videos marked as `DETECTED` or `NO_EVENT`
- Continues from where it left off
- Videos with `ERROR` status are retried

### Configuration

In `config.yaml`, customize the adverse event detection:

```yaml
adverse_event:
  model: "gemini-2.0-flash"       # Gemini model to use
  aggregate_file: "adverse_events_all.jsonl"
  log_file: "adverse_event_log.csv"
```

## ⚠️ Common Issues

- **FFmpeg Error**: If you see FFmpeg is not installed, you cannot extract audio. Install it via your OS package manager.
- **Sign-in Required**: If yt-dlp fails with "Sign in to confirm you're not a bot", ensure your `www.youtube.com_cookies.txt` is fresh and placed in the root folder.
- **GPU Memory**: If running locally with a small GPU, change `model_size` in `config.yaml` from `large` to `medium` or `small`.
- **API Rate Limits**: The adverse event detector includes a 30-second delay between API calls to respect rate limits. Adjust if needed.
- **Missing Config Sections**: If you skip certain steps (e.g., VLM or Adverse Event Detection), ensure those config sections exist in `config.yaml` or they will be skipped with a warning.
