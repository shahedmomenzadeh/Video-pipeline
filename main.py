# Entry point for the full pipeline
import yaml
import os
import sys
import argparse
from modules import downloader, cleaner, transcriber, refiner, summarizer, vlm_generator, adverse_event_detector

def load_config(config_path="config.yaml"):
    if not os.path.exists(config_path):
        print(f"Config file not found at {config_path}")
        sys.exit(1)
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def load_video_links(file_path):
    if not os.path.exists(file_path):
        print(f"⚠️ Warning: Video links file not found at {file_path}")
        return []
    with open(file_path, 'r') as f:
        links = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    print(f"ℹ️ Loaded {len(links)} URL(s) from {file_path}")
    return links

def main():
    # 1. Parse Command Line Arguments
    parser = argparse.ArgumentParser(description="Video Processing Pipeline")
    parser.add_argument(
        "--step", 
        type=str, 
        default="all", 
        choices=["all", "download", "clean", "transcribe", "refine", "summarize", "vlm", "adverse_event"],
        help="Specific pipeline step to run. Default is 'all' (runs sequentially)."
    )
    args = parser.parse_args()

    # 2. Load Config
    config = load_config()
    
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
    print(f"   VIDEO PIPELINE STARTED (Step: {args.step.upper()})")
    print("============================================")

    # Step 1: Downloader
    if args.step in ['all', 'download']:
        print("\n[Step 1/7] Running Downloader...")
        video_urls = load_video_links(links_file)
        if video_urls:
            downloader.run_downloader_pipeline(video_urls, videos_dir, audio_dir, metadata_file, cookies_file)
        else:
            print("❌ No URLs found. Skipping.")

    # Step 2: Cleaner
    if args.step in ['all', 'clean']:
        print("\n[Step 2/7] Running Cleaner...")
        cleaner.run_cleaner_pipeline(metadata_file, videos_dir, audio_dir, config['download']['max_duration_seconds'], auto_confirm=True)

    # Step 3: Transcriber
    if args.step in ['all', 'transcribe']:
        print("\n[Step 3/7] Running Whisper Transcriber...")
        transcriber.transcribe_audio_files(audio_dir, transcripts_dir, config['whisper']['model_size'], config['whisper']['device'])

    # Step 4: Refiner
    if args.step in ['all', 'refine']:
        print("\n[Step 4/7] Running LLM Refiner...")
        refiner.run_refiner_pipeline(transcripts_dir, videos_dir, refined_dir, log_file, config['cerebras']['model'], config['cerebras']['api_call_delay_seconds'], config['cerebras']['max_files_per_run'])

    # Step 5: Summarizer
    if args.step in ['all', 'summarize']:
        print("\n[Step 5/7] Creating Dataset Summary...")
        summarizer.create_dataset_info(metadata_file, log_file, summary_file)
    
    # Step 6: VLM Generator
    if args.step in ['all', 'vlm']:
        if 'vlm' in config:
            print("\n[Step 6/7] Generating VLM Fine-Tuning Dataset...")
            
            # New Output Directory
            vlm_dir = config['directories']['vlm_dataset']
            
            vlm_generator.run_vlm_generation_pipeline(
                dataset_summary_path=summary_file,
                refined_dir=refined_dir,
                output_dir=vlm_dir,
                aggregate_filename=config['vlm']['aggregate_file'],
                log_filename=config['vlm']['log_file'],
                gatekeeper_model=config['vlm']['gatekeeper_model'],
                generator_model=config['vlm']['generator_model']
            )
        else:
            print("\n⚠️ Skipping VLM Generation (Config missing)")

    # Step 7: Adverse Event Detector
    if args.step in ['all', 'adverse_event']:
        if 'adverse_event' in config:
            print("\n[Step 7/7] Running Adverse Event Detector...")
            adverse_event_detector.run_adverse_event_pipeline(
                vlm_input_dir=config['directories']['vlm_dataset'],
                output_dir=config['directories']['adverse_events'],
                log_filename=config['adverse_event']['log_file'],
                aggregate_filename=config['adverse_event']['aggregate_file'],
                model_name=config['adverse_event']['model']
            )
        else:
            print("\n⚠️ Skipping Adverse Event Detection (Config missing)")

    print("\n============================================")
    print("   PIPELINE COMPLETED SUCCESSFULLY")
    print("============================================")

if __name__ == "__main__":
    main()