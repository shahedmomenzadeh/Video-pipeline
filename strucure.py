import os

def create_file_if_missing(path, content=""):
    """Create a file only if it does NOT already exist."""
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"[CREATED] File: {path}")
    else:
        print(f"[SKIPPED] File already exists: {path}")

def ensure_dir_if_missing(path):
    """Create a directory only if it does NOT already exist."""
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        print(f"[CREATED] Directory: {path}")
    else:
        print(f"[SKIPPED] Directory already exists: {path}")

def main():
    # --- Root-level files ---
    create_file_if_missing("config.yaml", "# YAML configuration\n")
    create_file_if_missing("main.py", "# Entry point for the full pipeline\n")
    create_file_if_missing("videos_link.txt", "# Add YouTube URLs here, one per line\n")
    create_file_if_missing("requirements.txt", "# Add Python dependencies here\n")

    # --- Modules folder ---
    ensure_dir_if_missing("modules")

    create_file_if_missing("modules/__init__.py")
    create_file_if_missing("modules/downloader.py", "# Handles YouTube downloads & FFmpeg extraction\n")
    create_file_if_missing("modules/cleaner.py", "# Deletes files longer than the threshold\n")
    create_file_if_missing("modules/transcriber.py", "# Whisper AI transcription logic\n")
    create_file_if_missing("modules/refiner.py", "# LLM (Cerebras) text cleaning logic\n")
    create_file_if_missing("modules/summarizer.py", "# Merges logs into a final dataset CSV\n")

    # --- Auto-created data folders ---
    ensure_dir_if_missing("videos")
    create_file_if_missing("videos/video_metadata.csv", "filename,resolution,duration,download_time\n")

    ensure_dir_if_missing("audio")
    ensure_dir_if_missing("transcripts")

    ensure_dir_if_missing("refined_transcripts")
    ensure_dir_if_missing("refined_transcripts/full_responses")

    create_file_if_missing("refined_transcripts/refinement_log.csv", "file,word_count,tokens,time_ms\n")

    print("\nProject structure check complete.")

if __name__ == "__main__":
    main()
