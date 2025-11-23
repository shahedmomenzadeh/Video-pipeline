# Handles YouTube downloads & FFmpeg extraction
import yt_dlp
import os
import pandas as pd
from datetime import datetime
import shutil
import subprocess
import time

def is_ffmpeg_installed():
    """Check if FFmpeg is installed and available in the system's PATH."""
    return shutil.which("ffmpeg") is not None

def extract_audio_ffmpeg(video_filepath: str, audio_dir: str) -> str | None:
    """
    Extracts audio from a video file using FFmpeg, converting it to 16kHz mono WAV.
    """
    if not os.path.exists(video_filepath):
        print(f"❌ Error: Video file not found at {video_filepath}")
        return None

    try:
        video_basename = os.path.basename(video_filepath)
        video_name_no_ext = os.path.splitext(video_basename)[0]
        audio_filename = f"{video_name_no_ext}.wav"
        output_audio_path = os.path.join(audio_dir, audio_filename)

        print(f"🎵 Extracting audio from '{video_basename}'...")

        # Command to extract audio, convert to PCM 16-bit little-endian,
        # set sample rate to 16kHz, mono channel, and overwrite output
        command = [
            'ffmpeg', '-i', video_filepath, '-vn', '-acodec', 'pcm_s16le',
            '-ar', '16000', '-ac', '1', '-y', output_audio_path
        ]

        # Run ffmpeg, suppressing stdout and stderr to keep the log clean
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"✅ Audio extracted: {output_audio_path}")
        return audio_filename

    except subprocess.CalledProcessError:
        print(f"❌ FFmpeg error during audio extraction for {video_filepath}.")
        return None
    except Exception as e:
        print(f"❌ Unexpected error during audio extraction: {e}")
        return None

def download_video_and_extract_audio(video_url: str,
                                     output_dir: str,
                                     audio_dir: str,
                                     metadata_file: str,
                                     cookie_file: str | None = None):
    """
    Downloads a YouTube video, extracts its audio, logs metadata, and skips processed videos.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(audio_dir, exist_ok=True)

    # Define the metadata columns
    metadata_columns = [
        'title', 'channel_name', 'url', 'filename',
        'download_date', 'duration_seconds', 'resolution', 'audio_filename'
    ]

    # Load or initialize metadata DataFrame
    if os.path.exists(metadata_file):
        try:
            metadata_df = pd.read_csv(metadata_file)
            # Ensure all required columns exist
            for col in metadata_columns:
                if col not in metadata_df.columns:
                    metadata_df[col] = None
            # Reorder columns
            metadata_df = metadata_df[metadata_columns]
        except pd.errors.EmptyDataError:
            metadata_df = pd.DataFrame(columns=metadata_columns)
    else:
        metadata_df = pd.DataFrame(columns=metadata_columns)

    # Skip if video URL already processed
    if video_url in metadata_df['url'].values:
        print(f"⏩ Video already in metadata (skipped): {video_url}")
        return

    # yt-dlp options
    ydl_opts = {
        'format': 'bestvideo[height=480][ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
        'noplaylist': True,
        'merge_output_format': 'mp4',
        'postprocessors': [{'key': 'FFmpegMetadata', 'add_chapters': False}],
        'retries': 5,
        'fragment_retries': 5,
        'no_warnings': True,
    }

    if cookie_file and os.path.exists(cookie_file):
        ydl_opts['cookiefile'] = cookie_file

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Extract info first without downloading
            info = ydl.extract_info(video_url, download=False)
            video_title = str(info.get('title', 'Unknown Title'))
            channel_name = info.get('uploader', 'Unknown Channel')
            duration = info.get('duration')
            width, height = info.get('width'), info.get('height')
            resolution = f"{width}x{height}" if width and height else "N/A"

            # Get the expected downloaded video path
            expected_video_path = ydl.prepare_filename(info)
            video_name_no_ext = os.path.splitext(os.path.basename(expected_video_path))[0]
            expected_audio_filename = f"{video_name_no_ext}.wav"
            expected_audio_path = os.path.join(audio_dir, expected_audio_filename)

            # Skip if audio file already exists
            if os.path.exists(expected_audio_path):
                print(f"⏩ Audio already exists, assuming processed: {expected_audio_filename}")
                # Log metadata if it was missing
                if video_url not in metadata_df['url'].values:
                    new_entry = pd.DataFrame([{
                        'title': video_title,
                        'channel_name': channel_name,
                        'url': video_url,
                        'filename': os.path.basename(expected_video_path),
                        'download_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'duration_seconds': duration,
                        'resolution': resolution,
                        'audio_filename': expected_audio_filename
                    }])
                    metadata_df = pd.concat([metadata_df, new_entry], ignore_index=True)
                    metadata_df.to_csv(metadata_file, index=False)
                return

            print(f"⬇️ Downloading: '{video_title}' from channel: {channel_name}")
            ydl.download([video_url])

            # Verify download and extract audio
            if os.path.exists(expected_video_path):
                print(f"✅ Download complete: {os.path.basename(expected_video_path)}")
                audio_filename = extract_audio_ffmpeg(expected_video_path, audio_dir)

                # Log new entry to metadata
                new_entry = pd.DataFrame([{
                    'title': video_title,
                    'channel_name': channel_name,
                    'url': video_url,
                    'filename': os.path.basename(expected_video_path),
                    'download_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'duration_seconds': duration,
                    'resolution': resolution,
                    'audio_filename': audio_filename if audio_filename else "N/A"
                }])
                metadata_df = pd.concat([metadata_df, new_entry], ignore_index=True)
                metadata_df.to_csv(metadata_file, index=False)
            else:
                print(f"❌ Download reported success but file not found at '{expected_video_path}'")

    except yt_dlp.utils.DownloadError as e:
        print(f"❌ yt-dlp Download Error for {video_url}: {e}")
    except Exception as e:
        print(f"❌ Unexpected error for {video_url}: {e}")

def verify_and_process_existing_videos(videos_dir: str, audio_dir: str):
    """
    Scans the videos directory and extracts audio for any video missing its corresponding .wav file.
    """
    print("\n--- Verifying Existing Videos ---")
    if not os.path.isdir(videos_dir):
        print(f"❌ Verification skipped: '{videos_dir}' not found")
        return

    # Get a set of audio filenames (without extension)
    existing_audio_names = {os.path.splitext(f)[0] for f in os.listdir(audio_dir) if f.endswith('.wav')}
    video_files = [f for f in os.listdir(videos_dir) if f.endswith(('.mp4', '.mkv', '.webm', '.mov'))]

    # Find videos where the filename (without extension) is not in the audio set
    missing_audio_videos = [
        os.path.join(videos_dir, vf)
        for vf in video_files
        if os.path.splitext(vf)[0] not in existing_audio_names
    ]

    if not missing_audio_videos:
        print("✅ All videos have corresponding audio files.")
        return

    print(f"⚠️ Found {len(missing_audio_videos)} video(s) missing audio:")
    for v in missing_audio_videos:
        print(f"  - {os.path.basename(v)}")

    success, fail = 0, 0
    for vpath in missing_audio_videos:
        if extract_audio_ffmpeg(vpath, audio_dir):
            success += 1
        else:
            fail += 1

    print("\n--- Verification Summary ---")
    print(f"✅ Extracted: {success}")
    print(f"❌ Failed: {fail}")

def run_downloader_pipeline(urls: list, videos_dir: str, audio_dir: str, metadata_file: str, cookie_file: str):
    """
    Main entry point for the downloader module.
    """
    if not is_ffmpeg_installed():
        print("=" * 60)
        print("⚠️ FFmpeg is not installed or not in your system PATH.")
        return

    # Handle playlists vs single videos
    all_individual_urls = []
    info_opts = {
        'extract_flat': 'in_playlist',
        'skip_download': True,
        'quiet': True,
        'no_warnings': True,
    }
    if cookie_file and os.path.exists(cookie_file):
        info_opts['cookiefile'] = cookie_file

    print("Inspecting provided URLs for playlists...")
    with yt_dlp.YoutubeDL(info_opts) as ydl:
        for url in urls:
            print(f"Inspecting: {url}")
            try:
                info = ydl.extract_info(url, download=False)
                if info.get('_type') == 'playlist':
                    print(f"  -> 🔗 Found playlist: {info.get('title', 'Unknown Playlist')}")
                    playlist_video_urls = [entry.get('url') for entry in info.get('entries', []) if entry and entry.get('url')]
                    all_individual_urls.extend(playlist_video_urls)
                    print(f"  -> Added {len(playlist_video_urls)} videos from playlist.")
                else:
                    print("  -> Single video found.")
                    all_individual_urls.append(url)
            except Exception as e:
                print(f"  -> ❌ Error inspecting URL {url}: {e}")

    print(f"\n--- Total individual videos to process: {len(all_individual_urls)} ---")

    for i, video_url in enumerate(all_individual_urls):
        download_video_and_extract_audio(
            video_url,
            output_dir=videos_dir,
            audio_dir=audio_dir,
            metadata_file=metadata_file,
            cookie_file=cookie_file
        )
        if i < len(all_individual_urls) - 1:
            time.sleep(1)

    verify_and_process_existing_videos(videos_dir, audio_dir)