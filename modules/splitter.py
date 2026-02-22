import os
import json
import subprocess
import csv
import time
from tqdm import tqdm
import shutil

def is_ffmpeg_installed():
    """Check if FFmpeg is installed and available in the system's PATH."""
    return shutil.which("ffmpeg") is not None

def time_str_to_seconds(time_str):
    """Converts a time string like 'MM:SS' or 'HH:MM:SS' to total seconds."""
    if not time_str or "?" in time_str:
        return None
    
    parts = time_str.strip().split(':')
    try:
        if len(parts) == 2: # MM:SS
            return int(parts[0]) * 60 + int(parts[1])
        elif len(parts) == 3: # HH:MM:SS
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        else:
            return None
    except ValueError:
        return None

def find_video(videos_dir, original_filename, video_id):
    """Attempts to find the raw video file in the videos directory."""
    if original_filename:
        exact_path = os.path.join(videos_dir, original_filename)
        if os.path.exists(exact_path):
            return exact_path
            
    # Fallback to checking common extensions using the video_id
    for ext in ['.mp4', '.mkv', '.webm', '.mov']:
        path = os.path.join(videos_dir, f"{video_id}{ext}")
        if os.path.exists(path):
            return path
            
    return None

def setup_log_csv(log_path):
    """Creates the CSV log file with headers if it doesn't exist."""
    headers = ["video_id", "clips_generated", "errors", "timestamp"]
    if not os.path.exists(log_path):
        with open(log_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

def append_to_log_csv(log_path, video_id, clips_generated, errors):
    """Appends a single row to the CSV log."""
    with open(log_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            video_id,
            clips_generated,
            errors,
            time.strftime("%Y-%m-%d %H:%M:%S")
        ])

def extract_clip(input_video_path, output_video_path, start_sec, duration_sec):
    """
    Uses FFmpeg to accurately extract a clip from the parent video.
    Re-encodes video to ensure exact frame cuts, copies audio.
    """
    command = [
        'ffmpeg', '-y',
        '-ss', str(start_sec),
        '-i', input_video_path,
        '-t', str(duration_sec),
        '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '22',
        '-c:a', 'copy',
        output_video_path
    ]
    
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        return False
    except Exception as e:
        print(f"❌ Unexpected error during FFmpeg execution: {e}")
        return False

def run_splitter_pipeline(vlm_input_dir, videos_dir, output_dir, log_filename):
    print("\n=== VIDEO SPLITTER PIPELINE ===\n")

    if not is_ffmpeg_installed():
        print("❌ Error: FFmpeg is not installed or not in your system PATH. Required for splitting videos.")
        return

    # 1. Setup Directories and Files
    os.makedirs(output_dir, exist_ok=True)
    log_file_path = os.path.join(output_dir, log_filename)
    setup_log_csv(log_file_path)

    # 2. List Input Files (VLM JSONL files)
    if not os.path.exists(vlm_input_dir):
        print(f"❌ Input directory not found: {vlm_input_dir}")
        return

    # Only process individual video jsonl files, ignore the aggregate
    input_files = [f for f in os.listdir(vlm_input_dir) if f.endswith('.jsonl') and "all.jsonl" not in f]
    
    if not input_files:
        print("⚠️ No input VLM JSONL files found.")
        return

    print(f"Found {len(input_files)} VLM files to process for clipping.\n")

    total_clips_generated = 0
    total_videos_processed = 0

    for filename in tqdm(input_files, desc="Splitting Videos"):
        video_id = os.path.splitext(filename)[0]
        vlm_file_path = os.path.join(vlm_input_dir, filename)

        # Load VLM Data
        try:
            with open(vlm_file_path, 'r', encoding='utf-8') as f:
                line = f.readline()
                if not line:
                    continue
                vlm_data = json.loads(line)
        except Exception as e:
            tqdm.write(f"❌ Error reading {filename}: {e}")
            continue

        annotations = vlm_data.get("vlm_annotations", [])
        if not annotations:
            continue

        # Find the raw video file
        original_filename = vlm_data.get("original_filename")
        raw_video_path = find_video(videos_dir, original_filename, video_id)

        if not raw_video_path:
            tqdm.write(f"⚠️ Raw video not found for {video_id}. Searched in {videos_dir}.")
            append_to_log_csv(log_file_path, video_id, 0, "Video missing")
            continue

        # Create output directory for this specific video
        video_clip_dir = os.path.join(output_dir, video_id)
        os.makedirs(video_clip_dir, exist_ok=True)

        clips_created = 0
        clip_errors = 0

        # Process each step in the annotations
        for step in annotations:
            step_number = step.get("step_number")
            start_str = step.get("timestamp_start")
            end_str = step.get("timestamp_end")

            start_sec = time_str_to_seconds(start_str)
            end_sec = time_str_to_seconds(end_str)

            if start_sec is None or end_sec is None or start_sec >= end_sec:
                clip_errors += 1
                continue

            duration_sec = end_sec - start_sec
            clip_id = f"{video_id}_clip_{step_number:02d}"
            
            # File paths for the new clip
            clip_video_filename = f"clip_{step_number:02d}.mp4"
            clip_jsonl_filename = f"clip_{step_number:02d}.jsonl"
            
            clip_video_path = os.path.join(video_clip_dir, clip_video_filename)
            clip_jsonl_path = os.path.join(video_clip_dir, clip_jsonl_filename)

            # Skip if already processed
            if os.path.exists(clip_video_path) and os.path.exists(clip_jsonl_path):
                clips_created += 1
                continue

            # 1. Extract Video Clip using FFmpeg
            success = extract_clip(raw_video_path, clip_video_path, start_sec, duration_sec)
            
            if not success:
                clip_errors += 1
                continue

            # 2. Construct and Save Individual JSONL data
            clip_data = {
                "clip_id": clip_id,
                "parent_video_id": video_id,
                "clip_filename": clip_video_filename,
                "parent_video_title": vlm_data.get("video_title"),
                "parent_video_url": vlm_data.get("video_url"),
                "step_number": step_number,
                "clip_duration_seconds": duration_sec,
                "timestamp_start_in_parent": start_str,
                "timestamp_end_in_parent": end_str,
                "step_title": step.get("step_title"),
                "visual_description": step.get("visual_description"),
                "transcript_context": step.get("transcript_context"),
                "instruments": step.get("instruments", []),
                "anatomy": step.get("anatomy", [])
            }

            with open(clip_jsonl_path, 'w', encoding='utf-8') as f:
                f.write(json.dumps(clip_data) + "\n")
            
            clips_created += 1

        total_clips_generated += clips_created
        total_videos_processed += 1
        
        # Log completion for this video
        error_msg = f"{clip_errors} step(s) failed" if clip_errors > 0 else "None"
        append_to_log_csv(log_file_path, video_id, clips_created, error_msg)

    print("\n" + "="*30)
    print("   VIDEO SPLITTING COMPLETED")
    print("="*30)
    print(f"🎬 Videos Processed: {total_videos_processed}")
    print(f"✂️ Total Clips Made: {total_clips_generated}")
    print(f"📂 Output Folder:    {os.path.abspath(output_dir)}")