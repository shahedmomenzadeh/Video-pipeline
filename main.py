# Entry point for the full pipeline
import yaml
import os
import sys
from modules import downloader, cleaner, transcriber, refiner, summarizer

def load_config(config_path="config.yaml"):
    if not os.path.exists(config_path):
        print(f"Config file not found at {config_path}")
        sys.exit(1)
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def load_video_links(file_path):
    """Reads a text file and returns a list of non-empty URLs."""
    if not os.path.exists(file_path):
        print(f"⚠️ Warning: Video links file not found at {file_path}")
        return []
    
    with open(file_path, 'r') as f:
        # Read lines, strip whitespace, and ignore empty lines or comments
        links = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    print(f"ℹ️ Loaded {len(links)} URL(s) from {file_path}")
    return links

def main():
    # 1. Load Config
    config = load_config()
    
    # Path definitions from config
    videos_dir = config['directories']['videos']
    audio_dir = config['directories']['audio']
    transcripts_dir = config['directories']['transcripts']
    refined_dir = config['directories']['refined_transcripts']
    
    metadata_file = os.path.join(videos_dir, config['files']['metadata_csv'])
    cookies_file = config['files']['cookies']
    log_file = os.path.join(refined_dir, config['files']['refinement_log'])
    links_file = config['files']['video_links']
    summary_file = config['files']['dataset_summary']

    print("============================================")
    print("   VIDEO PROCESSING PIPELINE STARTED")
    print("============================================")

    # 2. Run Downloader & Audio Extractor
    print("\n[Step 1/5] Running Downloader...")
    
    video_urls = load_video_links(links_file)
    
    if video_urls:
        downloader.run_downloader_pipeline(
            urls=video_urls,
            videos_dir=videos_dir,
            audio_dir=audio_dir,
            metadata_file=metadata_file,
            cookie_file=cookies_file
        )
    else:
        print("❌ No URLs found to process. Skipping download step.")

    # 3. Run Cleaner (Remove long videos)
    print("\n[Step 2/5] Running Cleaner...")
    cleaner.run_cleaner_pipeline(
        metadata_file=metadata_file,
        videos_dir=videos_dir,
        audio_dir=audio_dir,
        max_duration_seconds=config['download']['max_duration_seconds'],
        auto_confirm=True 
    )

    # 4. Run Transcriber
    print("\n[Step 3/5] Running Whisper Transcriber...")
    transcriber.transcribe_audio_files(
        input_dir=audio_dir,
        output_dir=transcripts_dir,
        model_size=config['whisper']['model_size'],
        device=config['whisper']['device']
    )

    # 5. Run Refiner (LLM)
    print("\n[Step 4/5] Running LLM Refiner...")
    refiner.run_refiner_pipeline(
        input_dir=transcripts_dir,
        videos_dir=videos_dir,
        refined_output_dir=refined_dir,
        log_path=log_file,
        model_name=config['cerebras']['model'],
        delay=config['cerebras']['api_call_delay_seconds'],
        max_files=config['cerebras']['max_files_per_run']
    )

    # 6. Run Summarizer
    print("\n[Step 5/5] Creating Dataset Summary...")
    summarizer.create_dataset_info(
        metadata_path=metadata_file,
        log_path=log_file,
        output_file=summary_file
    )

    print("\n============================================")
    print("   PIPELINE COMPLETED SUCCESSFULLY")
    print("============================================")

if __name__ == "__main__":
    main()