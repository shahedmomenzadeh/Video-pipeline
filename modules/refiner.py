# LLM (Cerebras) text cleaning logic
import os
import json
import time
from tqdm import tqdm
import csv
from cerebras.cloud.sdk import Cerebras
from dotenv import load_dotenv

MEDICAL_EDITOR_SYSTEM_PROMPT = """You are an expert JSON and medical editor. Your task is to correct typos, punctuation, and grammatical errors in a JSON file provided by the user, while preserving its exact structure.

The user will provide a JSON array of segments from a cataract surgery video.
Your job is to fix errors **only** in the "text" fields.

**CRITICAL INSTRUCTIONS:**
1.  Read the user's JSON, perform your corrections, and think.
2.  You **MUST** return the JSON in the **EXACT** same array format, including "start", "end", and "text" keys for every segment.
3.  **DO NOT** alter the "start", "end", or any other part of the JSON structure.
4.  **DO NOT** include any commentary, conversational replies, or pre-amble.
5.  The output must be the pure, corrected JSON data and nothing else."""

def get_api_key():
    # Try loading from .env file locally
    load_dotenv()
    key = os.getenv("CEREBRAS_API_KEY")
    # Fallback for Colab env if needed, though this script is designed for local use now
    if not key:
        try:
            from google.colab import userdata
            key = userdata.get('CEREBRAS_API_KEY')
        except ImportError:
            pass
    return key

def parse_llm_json_output(raw_output: str) -> str:
    THINK_END_TAG = "</think>"
    if THINK_END_TAG in raw_output:
        after_think = raw_output.split(THINK_END_TAG, 1)[1].strip()
    else:
        after_think = raw_output.strip()

    start = after_think.find("[")
    end = after_think.rfind("]")

    if start != -1 and end != -1 and end > start:
        return after_think[start:end + 1].strip()
    return after_think

def format_time_to_mm_ss(seconds: float) -> str:
    s = int(seconds)
    return f"{s//60:02d}:{s%60:02d}"

def format_segments_to_txt(segments: list) -> str:
    lines = []
    for seg in segments:
        try:
            start = format_time_to_mm_ss(seg["start"])
            end = format_time_to_mm_ss(seg["end"])
            text = seg["text"].strip()
            lines.append(f"[{start} - {end}]: {text}")
        except KeyError:
            continue
    return "\n".join(lines)

def find_matching_video(videos_dir: str, basename: str) -> tuple[str, bool]:
    if not os.path.isdir(videos_dir):
        return "NOT_FOUND", False
    target = basename.lower()
    for filename in os.listdir(videos_dir):
        name, ext = os.path.splitext(filename)
        if name.lower() == target and ext.lower() in {'.mp4','.mov','.avi','.mkv','.webm','.flv','.m4v','.wmv','.mpeg','.mpg'}:
            return filename, True
    return "NOT_FOUND", False

def refine_with_llm(text: str, model_name: str, api_key: str, client) -> str | None:
    try:
        messages = [
            {"role": "system", "content": MEDICAL_EDITOR_SYSTEM_PROMPT},
            {"role": "user", "content": text}
        ]
        response = client.chat.completions.create(messages=messages, model=model_name)
        return response.choices[0].message.content
    except Exception as e:
        if any(x in str(e).lower() for x in ["context", "limit", "too large"]):
            tqdm.write(f"Context limit exceeded: {e}")
            return '{"error": "MODEL CONTEXT LIMIT EXCEEDED"}'
        tqdm.write(f"API error: {e} → retrying in 10s...")
        time.sleep(10)
        return None

def setup_csv_log(log_path):
    if os.path.exists(log_path):
        return
    with open(log_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["video_name", "transcript_name", "total_characters", "total_words", "video_found", "success"])

def log_result(log_path, row: list):
    with open(log_path, 'a', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow(row)

def run_refiner_pipeline(input_dir, videos_dir, refined_output_dir, log_path, model_name, delay, max_files):
    print("\n=== TRANSCRIPT REFINEMENT PIPELINE ===\n")
    full_response_dir = os.path.join(refined_output_dir, 'full_responses')
    
    os.makedirs(refined_output_dir, exist_ok=True)
    os.makedirs(full_response_dir, exist_ok=True)
    setup_csv_log(log_path)

    api_key = get_api_key()
    if not api_key:
        print("❌ CEREBRAS_API_KEY not found in environment variables or .env file!")
        return

    client = Cerebras(api_key=api_key)

    # Find files
    try:
        raw = {os.path.splitext(f)[0] for f in os.listdir(input_dir) if f.endswith('.json')}
        done = {os.path.splitext(f)[0] for f in os.listdir(refined_output_dir) if f.endswith('.txt')}
    except FileNotFoundError as e:
        print(f"Directory not found: {e}")
        return

    to_process = sorted(raw - done)[:max_files]
    if not to_process:
        print("✅ All transcripts already refined!")
        return

    print(f"Processing {len(to_process)} transcript(s)...\n")

    for basename in tqdm(to_process, desc="Refining"):
        video_name, video_found = find_matching_video(videos_dir, basename)
        
        json_path = os.path.join(input_dir, f"{basename}.json")
        txt_path = os.path.join(refined_output_dir, f"{basename}.txt")
        full_path = os.path.join(full_response_dir, f"{basename}_full_response.txt")

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                segments = json.load(f)
        except Exception as e:
            tqdm.write(f"Failed to load {basename}.json → {e}")
            continue

        if not segments:
            continue

        payload = json.dumps(segments, indent=4)
        
        raw_response = None
        while raw_response is None:
            raw_response = refine_with_llm(payload, model_name, api_key, client)

        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(raw_response)

        success = False
        total_chars = total_words = 0
        try:
            cleaned_json = parse_llm_json_output(raw_response)
            refined_segments = json.loads(cleaned_json)
            total_words = sum(len(seg.get("text", "").split()) for seg in refined_segments)
            formatted_text = format_segments_to_txt(refined_segments)
            total_chars = len(formatted_text)

            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(formatted_text)
            success = True
        except Exception as e:
            tqdm.write(f"Failed to parse/save {basename} → {e}")

        log_result(log_path, [video_name, f"{basename}.txt", total_chars, total_words, video_found, success])

        if basename != to_process[-1]:
            time.sleep(delay)

    print("\nRefinement process completed.")