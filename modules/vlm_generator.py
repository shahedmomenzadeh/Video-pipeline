import os
import json
import pandas as pd
from tqdm import tqdm
from google import genai
from google.genai import types
from dotenv import load_dotenv
import csv
import time

# --- PROMPTS ---

GATEKEEPER_PROMPT = """Role: You are a Surgical Data Curator.
Task: Analyze the provided video transcript and determine if the corresponding video is suitable for a surgical instruction dataset.
Evaluation Criteria:

Discard (Label: NO):
The transcript is empty or contains only non-verbal markers (e.g., [Music], [Silence], [Applause]).
The text is incoherent, gibberish, or appears to be a "hallucination" from the speech-to-text engine.
The content is purely conversational (e.g., talking about lunch, scheduling) with no medical terminology.

Keep (Label: YES):
The transcript contains specific medical terminology (anatomy, instruments, pathology).
The speaker is narrating actions, steps, or educational concepts related to surgery.

Input Transcript:
{transcript_text}

Output format:
Provide your response in JSON format only:
JSON
{{
  "decision": "YES", // or "NO"
  "confidence_score": 0.0 to 1.0,
  "reasoning": "Brief explanation..."
}}"""

ANALYST_PROMPT = """Role: You are an expert Ophthalmic Surgical Analyst and Data Annotator (specialized in cataract surgery).
Task: Analyze the provided operating-microscope surgical video stream (primary microscope view) and the accompanying transcript. Your goal is to generate a structured dataset entry for fine-tuning a Vision–Language Model to understand cataract surgery workflows and visuals.
Instructions:
Visual priority

Prioritize what is visually occurring in the microscope video. Use the transcript only to support context, identify terminology, or disambiguate instruments when necessary.
If audio/surgeon commentary is out of sync with the video, follow the video timing and actions.
Prefer the main microscope view when multiple views exist. Note overlays (timer, phaco settings) only if they help timestamp steps.

Segmentation

Break the video into distinct surgical steps (events). Typical cataract step boundaries include (but are not limited to): conjunctival/limbal prep, corneal/limbal incision, creation of paracentesis, injection of viscoelastic, continuous curvilinear capsulorhexis (CCC), hydrodissection/hydrodelineation, phacoemulsification (sculpting/chopping/aspiration), cortical irrigation/aspiration (I/A), intraocular lens (IOL) insertion (foldable injector), viscoelastic removal/AC reformation, wound hydration/closure, and any complication management (posterior capsule rupture, iris prolapse, etc.).
Label the start and end of each step based on visible actions (e.g., first visible entry of a keratome into cornea marks incision start; completion is when the wound is formed and instrument withdrawn).

Granularity

For each step, explicitly identify:
The specific instrument(s) used (e.g., 2.2 mm keratome, cystotome, capsulorhexis forceps, phaco handpiece/phaco probe, chopper, irrigation/aspiration (I/A) cannula, viscoelastic cannula, IOL injector, Sinskey hook, microforceps).
The anatomical structure(s) being manipulated (e.g., cornea — clear corneal incision, anterior chamber, anterior capsule, lens nucleus, lens cortex, capsular bag, posterior capsule, iris, zonules, sclera).
The surgical sub-action or maneuver when relevant (e.g., "capsulorhexis initiated with cystotome and completed with forceps," "nuclear chopping using vertical chop technique," "hydrodissection bubble separation visible").

Format

Output the result strictly in the JSON array format defined below. Do not add extra fields, comments, or trailing text outside the JSON structure.
Use timestamp strings in "MM:SS" format (or "HH:MM:SS" if the video is longer than one hour) and make sure timestamps reflect the microscope video timeline.
If a field has no applicable transcript quote, set "transcript_context": "".

Output Format (JSON):
[
{{
"step_number": 1,
"timestamp_start": "MM:SS",
"timestamp_end": "MM:SS",
"step_title": "Short title of the surgical action",
"visual_description": "Detailed description of the visual action.",
"transcript_context": "Relevant quote from transcript (if any)",
"instruments": ["List", "of", "instruments"],
"anatomy": ["List", "of", "anatomy"]
}}
]

Transcript: {transcript_text}"""


def get_gemini_client():
    """Initializes the Google GenAI client."""
    # Try getting key from Colab secrets first
    try:
        from google.colab import userdata
        api_key = userdata.get('GEMINI_API_KEY')
    except ImportError:
        api_key = None
    except Exception:
        api_key = None

    # Fallback to env variable
    if not api_key:
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        print("❌ Error: GEMINI_API_KEY not found. Please set it in .env or Colab Secrets.")
        return None
        
    return genai.Client(api_key=api_key)

def check_transcript_quality(client, model_name, transcript_text):
    """
    Step 1: Gatekeeper
    Uses a fast model to decide if the transcript is worth processing.
    """
    try:
        # Truncate transcript to avoid token limits on the gatekeeper if extremely long.
        prompt = GATEKEEPER_PROMPT.format(transcript_text=transcript_text[:25000])
        
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )
        return json.loads(response.text)
    except Exception as e:
        return {"decision": "ERROR", "reasoning": str(e)}

def generate_vlm_entry(client, model_name, video_url, transcript_text):
    """
    Step 2: Generator
    Uses a powerful model to watch the video (via URL) and analyze it with the transcript.
    """
    try:
        formatted_prompt = ANALYST_PROMPT.format(transcript_text=transcript_text)
        
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part(file_data=types.FileData(file_uri=video_url)),
                    types.Part.from_text(text=formatted_prompt),
                ],
            ),
        ]

        generate_content_config = types.GenerateContentConfig(
            response_mime_type="application/json",
            # thinking_config=types.ThinkingConfig(thinking_budget=1024) 
        )

        response = client.models.generate_content(
            model=model_name,
            contents=contents,
            config=generate_content_config,
        )
        
        return json.loads(response.text)

    except Exception as e:
        print(f"  ⚠️ VLM Generation failed for {video_url}: {e}")
        return None

def setup_log_csv(log_path):
    """Creates the CSV log file with headers if it doesn't exist."""
    headers = [
        "video_id", "original_filename", "status", "decision", "confidence", "reasoning", 
        "video_title", "url", "download_date", "timestamp"
    ]
    if not os.path.exists(log_path):
        with open(log_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

def append_to_log_csv(log_path, data):
    """Appends a single row to the CSV log."""
    with open(log_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            data.get("video_id"),
            data.get("original_filename"),
            data.get("status"),
            data.get("decision"),
            data.get("confidence"),
            data.get("reasoning"),
            data.get("video_title"),
            data.get("url"),
            data.get("download_date"),
            time.strftime("%Y-%m-%d %H:%M:%S")
        ])

def get_stable_id(filename_or_name):
    """
    Generates a stable ID by removing extensions.
    Example: 'my_video.mp4' -> 'my_video'
    """
    if not filename_or_name:
        return None
    return os.path.splitext(str(filename_or_name))[0]

def run_vlm_generation_pipeline(dataset_summary_path, refined_dir, output_dir, aggregate_filename, log_filename, gatekeeper_model, generator_model):
    print("\n=== VLM DATASET GENERATION PIPELINE ===\n")

    client = get_gemini_client()
    if not client:
        return

    # 1. Setup Directories and Files
    os.makedirs(output_dir, exist_ok=True)
    
    aggregate_file_path = os.path.join(output_dir, aggregate_filename)
    log_file_path = os.path.join(output_dir, log_filename)
    
    setup_log_csv(log_file_path)

    # 2. Load the dataset summary
    if not os.path.exists(dataset_summary_path):
        print(f"❌ Summary file not found at: {dataset_summary_path}")
        return
    
    df = pd.read_csv(dataset_summary_path)
    if df.empty:
        print("⚠️ Summary file is empty.")
        return
    
    # 3. Resume Logic
    # We look for files in the output directory that match {stable_id}.jsonl
    # This is more robust than just checking the log.
    existing_files = set(os.listdir(output_dir))
    
    # We also check the CSV log to see if we explicitly REJECTED/SKIPPED specific IDs before
    # FIX: We only count it as "processed" if the status is ACCEPTED or REJECTED.
    # If the status is ERROR_GENERATION, we want to retry it.
    processed_log_ids = set()
    if os.path.exists(log_file_path):
        try:
            log_df = pd.read_csv(log_file_path)
            # Filter rows where the job was truly finished (Successful or permanently Rejected)
            finished_df = log_df[log_df['status'].isin(['ACCEPTED', 'REJECTED'])]
            processed_log_ids = set(finished_df['video_id'].astype(str))
            print(f"ℹ️ Resuming... {len(processed_log_ids)} completed videos found in log.")
        except Exception:
            pass

    success_count = 0
    skip_count = 0
    error_count = 0
    already_done_count = 0

    print(f"Processing {len(df)} videos from summary...\n")

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Generating VLM Data"):
        
        raw_video_name = str(row.get('video_name', ''))
        video_url = row.get('url', '')
        video_title = row.get('title', '')
        download_date = row.get('download_date', '')
        
        if not raw_video_name or not video_url:
            continue

        # --- ROBUST ID GENERATION ---
        # Normalize the ID: "video.mp4" -> "video"
        video_id = get_stable_id(raw_video_name)
        
        # --- CHECK 1: File Existence ---
        # If video.jsonl exists, we are done.
        expected_filename = f"{video_id}.jsonl"
        if expected_filename in existing_files:
            # tqdm.write(f"  ⏩ Skipping {video_id} (File exists)")
            already_done_count += 1
            continue

        # --- CHECK 2: Log History ---
        # If we logged this ID as successfully processed or definitively rejected, skip.
        if video_id in processed_log_ids:
            # tqdm.write(f"  ⏩ Skipping {video_id} (Found in log)")
            already_done_count += 1
            continue

        # Locate transcript input
        transcript_name = str(row.get('transcript_name', ''))
        # Try finding the transcript using various naming conventions
        # 1. Exact match from CSV
        t_path_1 = os.path.join(refined_dir, transcript_name)
        # 2. Replace extension with .txt
        t_path_2 = os.path.join(refined_dir, f"{os.path.splitext(transcript_name)[0]}.txt")
        # 3. Use video_id + .txt
        t_path_3 = os.path.join(refined_dir, f"{video_id}.txt")

        if os.path.exists(t_path_1):
            final_t_path = t_path_1
        elif os.path.exists(t_path_2):
            final_t_path = t_path_2
        elif os.path.exists(t_path_3):
            final_t_path = t_path_3
        else:
            # If transcript is missing, we can't process
            error_count += 1
            continue

        try:
            with open(final_t_path, 'r', encoding='utf-8') as f:
                transcript_text = f.read()
        except Exception as e:
            error_count += 1
            continue

        # --- STEP 1: GATEKEEPER ---
        quality_result = check_transcript_quality(client, gatekeeper_model, transcript_text)
        decision = quality_result.get('decision', 'NO').upper()
        reason = quality_result.get('reasoning', 'No reason provided')
        confidence = quality_result.get('confidence_score', 0.0)
        
        if decision != 'YES':
            skip_count += 1
            # Log failure
            append_to_log_csv(log_file_path, {
                "video_id": video_id,
                "original_filename": raw_video_name,
                "status": "REJECTED",
                "decision": decision,
                "confidence": confidence,
                "reasoning": reason,
                "video_title": video_title,
                "url": video_url,
                "download_date": download_date
            })
            continue

        # --- STEP 2: GENERATOR ---
        tqdm.write(f"  🎥 Analyze: {video_title}")
        vlm_data = generate_vlm_entry(client, generator_model, video_url, transcript_text)
        sleep_time = 60 # seconds
        tqdm.write(f" ⏳ Waiting {sleep_time}s to avoid rate limits...")
        time.sleep(sleep_time)
        if vlm_data:
            final_entry = {
                "video_id": video_id,
                "original_filename": raw_video_name,
                "status": "SUCCESS",
                "video_url": video_url,
                "video_title": video_title,
                "download_date": download_date,
                "transcript_quality_check": quality_result,
                "vlm_annotations": vlm_data
            }

            # 1. Save INDIVIDUAL JSONL using stable ID
            individual_file_path = os.path.join(output_dir, expected_filename)
            with open(individual_file_path, 'w', encoding='utf-8') as f:
                f.write(json.dumps(final_entry) + "\n")

            # 2. Append to AGGREGATE JSONL
            with open(aggregate_file_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(final_entry) + "\n")
            
            # 3. Log SUCCESS
            append_to_log_csv(log_file_path, {
                "video_id": video_id,
                "original_filename": raw_video_name,
                "status": "ACCEPTED",
                "decision": decision,
                "confidence": confidence,
                "reasoning": reason,
                "video_title": video_title,
                "url": video_url,
                "download_date": download_date
            })

            success_count += 1
        else:
            error_count += 1
            tqdm.write(f"  ❌ Failed to generate VLM data for {video_id}")
            # Log ERROR
            append_to_log_csv(log_file_path, {
                "video_id": video_id,
                "original_filename": raw_video_name,
                "status": "ERROR_GENERATION",
                "decision": decision,
                "confidence": confidence,
                "reasoning": "Model failed to generate valid JSON",
                "video_title": video_title,
                "url": video_url,
                "download_date": download_date
            })

    print("\n" + "="*30)
    print("   VLM GENERATION COMPLETED")
    print("="*30)
    print(f"✅ Accepted & Generated: {success_count}")
    print(f"⏩ Rejected (Quality):   {skip_count}")
    print(f"⏭️ Skipped (Already Done): {already_done_count}")
    print(f"❌ Errors:               {error_count}")
    print(f"📂 Output Folder:        {os.path.abspath(output_dir)}")