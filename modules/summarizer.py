# Merges logs into a final dataset CSV
import pandas as pd
import os

def create_dataset_info(metadata_path, log_path, output_file):
    print("=== CREATING DATASET INFO ===\n")

    # 1. Check if files exist
    if not os.path.exists(metadata_path):
        print(f"❌ Error: Metadata file not found at {metadata_path}")
        return
    if not os.path.exists(log_path):
        print(f"❌ Error: Log file not found at {log_path}")
        return

    # 2. Load DataFrames
    try:
        df_meta = pd.read_csv(metadata_path)
        df_log = pd.read_csv(log_path)
        print("✅ Files loaded successfully.")
    except Exception as e:
        print(f"❌ Error reading CSV files: {e}")
        return

    # 3. Merge DataFrames
    # We join where log['video_name'] matches metadata['filename']
    print("Merging data...")
    # Ensure string types for merging keys to avoid mismatches
    df_log['video_name'] = df_log['video_name'].astype(str)
    df_meta['filename'] = df_meta['filename'].astype(str)

    merged_df = pd.merge(
        df_log,
        df_meta,
        left_on='video_name',
        right_on='filename',
        how='inner'
    )

    if merged_df.empty:
        print("⚠️ Warning: The merged dataset is empty. Check if filenames match between the two CSVs.")
        return

    # 4. Create Final DataFrame with requested columns
    final_df = pd.DataFrame()

    # Use .get() or direct access if you are sure columns exist. 
    # Using direct access based on your notebook code, but adding checks is safer.
    final_df['title'] = merged_df['title']
    final_df['video_name'] = merged_df['video_name']
    final_df['transcript_name'] = merged_df['transcript_name']
    
    # Mapping 'audio_filename' from metadata to 'audio_name'
    final_df['audio_name'] = merged_df['audio_filename']

    # Taking duration from metadata
    final_df['duration_seconds'] = merged_df['duration_seconds']

    # Mapping 'total_words' from log to 'word_count'
    final_df['word_count'] = merged_df['total_words']

    final_df['channel_name'] = merged_df['channel_name']
    final_df['url'] = merged_df['url']

    # Mapping 'filename' from metadata to 'download_name'
    final_df['download_name'] = merged_df['filename']

    # Mapping 'download_date'
    final_df['download_date'] = merged_df['download_date']

    # 5. Save to CSV
    try:
        final_df.to_csv(output_file, index=False)
        print("\n" + "="*30)
        print("       SUCCESS       ")
        print("="*30)
        print(f"Merged {len(final_df)} records.")
        print(f"Saved to: {os.path.abspath(output_file)}")
    except Exception as e:
        print(f"❌ Error saving summary file: {e}")