# Whisper AI transcription logic
import os
import whisper
import torch
import json
from tqdm import tqdm

def transcribe_audio_files(input_dir: str, output_dir: str, model_size: str = 'large', device: str = None):
    """
    Transcribes all .wav files in the input directory using Whisper.
    """
    print("--- Starting Audio Transcription Process ---")

    # 1. Setup directories and check for GPU
    os.makedirs(output_dir, exist_ok=True)
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    print(f"Using device: {device.upper()}")
    if device == 'cpu':
        print("⚠️ WARNING: No GPU found. Transcription will be very slow.")

    # 2. Load the pre-trained Whisper model
    print(f"Loading Whisper model ({model_size})...")
    try:
        model = whisper.load_model(model_size, device=device)
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading Whisper model: {e}")
        return

    # 3. Identify audio files to process
    if not os.path.exists(input_dir):
        print(f"❌ Input directory not found: {input_dir}")
        return

    audio_files = {os.path.splitext(f)[0] for f in os.listdir(input_dir) if f.endswith('.wav')}
    transcribed_files = {os.path.splitext(f)[0] for f in os.listdir(output_dir) if f.endswith('.json')}
    files_to_process = sorted([f + '.wav' for f in (audio_files - transcribed_files)])

    if not files_to_process:
        print("✅ All audio files have already been transcribed.")
        return

    print(f"Found {len(files_to_process)} audio file(s) to transcribe.")

    # 4. Process each audio file
    for filename in tqdm(files_to_process, desc="Transcribing Audio"):
        input_path = os.path.join(input_dir, filename)
        output_filename = f"{os.path.splitext(filename)[0]}.json"
        output_path = os.path.join(output_dir, output_filename)

        try:
            # Perform transcription
            result = model.transcribe(input_path, fp16=torch.cuda.is_available())

            # Extract segments
            segments = [
                {
                    "start": round(seg["start"], 1),
                    "end": round(seg["end"], 1),
                    "text": seg["text"].strip()
                }
                for seg in result["segments"]
            ]

            # Save JSON
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(segments, f, indent=4, ensure_ascii=False)

        except Exception as e:
            tqdm.write(f"❌ Error transcribing {filename}: {e}")

    print("\n--- Audio Transcription process completed. ---")