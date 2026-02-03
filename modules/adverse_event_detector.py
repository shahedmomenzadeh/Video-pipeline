import os
import json
import pandas as pd
from tqdm import tqdm
from google import genai
from google.genai import types
from dotenv import load_dotenv
import csv
import time

# --- PROMPT DEFINITION ---

SAFETY_ANALYST_PROMPT = """Role: You are a Surgical Safety Analyst specializing in Cataract Surgery complications.
Task: Analyze the provided chronological list of surgical steps (visual descriptions) to detect any intraoperative adverse events or complications.

Input Data:
You will receive a list of steps. Each step has a timestamp and a visual description of the surgical action derived from the video.

Adverse Events to Look For:
You must strictly identify only the following complications based on the visual descriptions provided:

1. Intra-operative Complications:
   - Iris Prolapse: Protrusion of iris tissue through the surgical wound (main or side port).
   - Zonular Dialysis: Partial or complete rupture of zonular fibers, leading to lens instability, equator visibility, or decentration.
   - IFIS (Intraoperative Floppy Iris Syndrome): Look for triad of signs: billowing iris stroma, iris prolapse, or progressive miosis (constriction).
   - Phaco Wound Burn: Thermal injury at the corneal incision characterized by whitening or graying of the wound edges.
   - Posterior Capsule Rupture (PCR): Breach of the posterior capsule. Look for "vitreous loss", "vitreous prolapse", "sulcus placement of IOL", "anterior vitrectomy", or "capsule tear".
   - Vitreous Loss: Vitreous humor entering the anterior chamber or exiting the eye (usually secondary to PCR).
   - Nucleus Drop: Lens nucleus or large fragments falling posteriorly into the vitreous cavity.
   - IOL Drop: Dislocation of the intraocular lens (IOL) into the vitreous.

2. Retinal / Posterior Segment Complications:
   - Peripheral Retinal Tear: Visible break in the peripheral retina (may be mentioned if view extends to fundus or red reflex changes significantly).
   - Retinal Hemorrhage: Bleeding visible in the posterior segment or retina.

Instructions:
- Analyze the text strictly. Do not hallucinate events not described.
- Infer complications from management actions: If the text describes "anterior vitrectomy", "sulcus IOL implantation", or "limbal approach for vitrectomy", you MUST infer that a Posterior Capsule Rupture (PCR) or Vitreous Loss occurred, even if the rupture itself wasn't explicitly described.
- If NO adverse events are found, return an empty list for "adverse_events".

Output Format:
Return strictly a JSON object with the following structure:
{
  "adverse_events": [
    {
      "event_name": "Name of the complication (e.g., Posterior Capsule Rupture)",
      "timestamp_start": "MM:SS (start of the event context)",
      "timestamp_end": "MM:SS (end of the event context)",
      "reason": "Quote or explanation from the visual description that indicates this event."
    }
  ]
}
"""

def get_gemini_client():
    """Initializes the Google GenAI client."""
    try:
        from google.colab import userdata
        api_key = userdata.get('GEMINI_API_KEY')
    except ImportError:
        api_key = None
    except Exception:
        api_key = None

    if not api_key:
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        print("❌ Error: GEMINI_API_KEY not found.")
        return None
        
    return genai.Client(api_key=api_key)

def setup_log_csv(log_path):
    """Creates the CSV log file with headers if it doesn't exist."""
    headers = ["video_id", "status", "event_count", "timestamp"]
    if not os.path.exists(log_path):
        with open(log_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

def append_to_log_csv(log_path, video_id, status, event_count):
    """Appends a single row to the CSV log."""
    with open(log_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            video_id,
            status,
            event_count,
            time.strftime("%Y-%m-%d %H:%M:%S")
        ])

def extract_visual_steps(vlm_data):
    """
    Formats the VLM annotations into a readable text format for the LLM.
    """
    annotations = vlm_data.get("vlm_annotations", [])
    if not annotations:
        return ""
    
    text_context = "Surgical Steps Timeline:\n"
    for step in annotations:
        start = step.get("timestamp_start", "??:??")
        end = step.get("timestamp_end", "??:??")
        desc = step.get("visual_description", "")
        text_context += f"[{start} - {end}]: {desc}\n"
    
    return text_context

def detect_adverse_events(client, model_name, visual_context_text):
    """
    Sends the visual descriptions to the LLM to find adverse events.
    """
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=SAFETY_ANALYST_PROMPT),
                        types.Part.from_text(text=f"\nAnalyze this surgery:\n{visual_context_text}")
                    ]
                )
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )
        return json.loads(response.text)
    except Exception as e:
        print(f"  ⚠️ Adverse Event Detection failed: {e}")
        return None

def run_adverse_event_pipeline(vlm_input_dir, output_dir, log_filename, aggregate_filename, model_name):
    print("\n=== ADVERSE EVENT DETECTION PIPELINE ===\n")

    client = get_gemini_client()
    if not client:
        return

    # 1. Setup Directories and Files
    os.makedirs(output_dir, exist_ok=True)
    log_file_path = os.path.join(output_dir, log_filename)
    aggregate_file_path = os.path.join(output_dir, aggregate_filename)
    
    setup_log_csv(log_file_path)

    # 2. Resuming Logic
    # We check the log to see which videos are already fully processed (DETECTED or NO_EVENT)
    processed_ids = set()
    if os.path.exists(log_file_path):
        try:
            log_df = pd.read_csv(log_file_path)
            # We skip anything that was successfully checked (whether events were found or not)
            finished_df = log_df[log_df['status'].isin(['DETECTED', 'NO_EVENT'])]
            processed_ids = set(finished_df['video_id'].astype(str))
            print(f"ℹ️ Resuming... {len(processed_ids)} videos already checked.")
        except Exception:
            pass

    # 3. List Input Files (VLM JSONL files)
    # We look for individual JSONL files in the vlm_dataset folder
    if not os.path.exists(vlm_input_dir):
        print(f"❌ Input directory not found: {vlm_input_dir}")
        return

    # Filter for .jsonl files but ignore the aggregate file from the previous step if it exists
    input_files = [f for f in os.listdir(vlm_input_dir) if f.endswith('.jsonl') and "all.jsonl" not in f]
    
    if not input_files:
        print("⚠️ No input VLM JSONL files found.")
        return

    print(f"Processing {len(input_files)} VLM files for safety analysis...\n")

    detected_count = 0
    clean_count = 0
    error_count = 0
    skipped_count = 0

    for filename in tqdm(input_files, desc="Scanning for Adverse Events"):
        video_id = os.path.splitext(filename)[0]

        # --- CHECK 1: Already Processed? ---
        if video_id in processed_ids:
            skipped_count += 1
            continue

        file_path = os.path.join(vlm_input_dir, filename)
        
        # Load VLM Data
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # We assume the file has one JSON object per line, but for individual files usually just one line
                line = f.readline()
                if not line:
                    continue
                vlm_data = json.loads(line)
        except Exception as e:
            tqdm.write(f"❌ Error reading {filename}: {e}")
            error_count += 1
            continue

        # Extract Context
        visual_context = extract_visual_steps(vlm_data)
        if not visual_context:
            tqdm.write(f"⚠️ No visual steps found for {video_id}, skipping.")
            continue

        # --- STEP 2: LLM ANALYSIS ---

        
        result = detect_adverse_events(client, model_name, visual_context)
        # 30s delay to be safe with rate limits
        time.sleep(30)

        if result is None:
            # API Error
            append_to_log_csv(log_file_path, video_id, "ERROR", 0)
            error_count += 1
            continue

        events = result.get("adverse_events", [])
        
        if events:
            # === ADVERSE EVENTS DETECTED ===
            detected_count += 1
            
            # Construct Output Object
            # "I want the format of the first part of jsonl files be the same as the ones in vlm_dataset"
            output_entry = {
                "video_id": vlm_data.get("video_id"),
                "original_filename": vlm_data.get("original_filename"),
                "status": "DETECTED", # Updated status
                "video_url": vlm_data.get("video_url"),
                "video_title": vlm_data.get("video_title"),
                "download_date": vlm_data.get("download_date"),
                # We do NOT include the full 'vlm_annotations' to keep the focus on events,
                # but we include the new events list.
                "adverse_events": events
            }

            # 1. Save INDIVIDUAL JSONL
            individual_output_path = os.path.join(output_dir, f"{video_id}.jsonl")
            with open(individual_output_path, 'w', encoding='utf-8') as f:
                f.write(json.dumps(output_entry) + "\n")

            # 2. Append to AGGREGATE JSONL
            with open(aggregate_file_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(output_entry) + "\n")
            
            # 3. Log
            append_to_log_csv(log_file_path, video_id, "DETECTED", len(events))
            tqdm.write(f"  🚨 Adverse Event Detected in {video_id}: {len(events)} event(s)")

        else:
            # === NO EVENTS (CLEAN) ===
            clean_count += 1
            # We do NOT save a JSONL file, but we MUST update the log so we don't check again.
            append_to_log_csv(log_file_path, video_id, "NO_EVENT", 0)

    print("\n" + "="*30)
    print("   SAFETY ANALYSIS COMPLETED")
    print("="*30)
    print(f"🚨 Events Detected:  {detected_count}")
    print(f"✅ Clean Videos:     {clean_count}")
    print(f"⏭️ Skipped (Done):   {skipped_count}")
    print(f"❌ Errors:           {error_count}")
    print(f"📂 Output Folder:    {os.path.abspath(output_dir)}")