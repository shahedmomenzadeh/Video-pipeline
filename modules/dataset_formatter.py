import os
import json
import csv
import time
import random
from tqdm import tqdm
from openai import OpenAI
from pydantic import BaseModel, ValidationError

# ==========================================
# 1. Define Pydantic Schemas for Output Verification
# ==========================================
class QAPair(BaseModel):
    question_type: str
    question: str
    options: dict[str, str]
    correct_answer: str
    reference_reasoning: str

class QAList(BaseModel):
    qa_pairs: list[QAPair]

# ==========================================
# 2. Prompts
# ==========================================
SYSTEM_PROMPT = """You are an expert surgical educator creating multiple-choice questions for a Vision-Language Model training dataset based on cataract surgery video clips.

Your task is to generate high-quality questions that test understanding of surgical procedures, visual observations, and instrument identification.

CRITICAL RULES:
1. Write naturally as if you're watching the video directly - NEVER mention metadata field names like "visual_description", "step_title", "instruments list", etc.
2. In your reasoning, describe what is visible in the video using natural language (e.g., "In the video, the surgeon uses sharp tips to puncture..." NOT "The visual_description states...")
3. Generate EXACTLY 3 questions covering these categories:
   - "step_identification": Identify the surgical step being performed
   - "visual_observation": Ask about specific visual details, techniques, or observations
   - "instrument_identification": Identify surgical tools being used

Each question object MUST include ALL of these fields:
- "question_type": one of ["step_identification", "visual_observation", "instrument_identification"]
- "question": the question text
- "options": a dictionary with exactly 4 keys: "A", "B", "C", and "D"
- "correct_answer": just the letter (e.g., "A")
- "reference_reasoning": natural explanation as if describing what you see in the video

Example format:
{
  "qa_pairs": [
    {
      "question_type": "step_identification",
      "question": "What surgical step is being performed in this clip?",
      "options": {
        "A": "Capsulorhexis",
        "B": "Phacoemulsification",
        "C": "IOL insertion",
        "D": "Corneal incision"
      },
      "correct_answer": "A",
      "reference_reasoning": "In this video clip, the surgeon is creating a circular opening in the anterior capsule of the lens, which is the defining characteristic of capsulorhexis."
    }
  ]
}
"""

# ==========================================
# 3. Helper Functions
# ==========================================
def setup_log_csv(log_path):
    headers = ["clip_id", "status", "timestamp"]
    if not os.path.exists(log_path):
        with open(log_path, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(headers)

def log_progress(log_path, clip_id, status):
    with open(log_path, 'a', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow([clip_id, status, time.strftime("%Y-%m-%d %H:%M:%S")])

def get_processed_clips(log_path):
    processed = set()
    if os.path.exists(log_path):
        with open(log_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['status'] == 'SUCCESS':
                    processed.add(row['clip_id'])
    return processed

def manage_train_val_test_split(clips_dir, output_dir, split_filename):
    """Ensures parent-level splitting to prevent data leakage."""
    split_filepath = os.path.join(output_dir, split_filename)
    
    # Load existing split if it exists (for resuming)
    if os.path.exists(split_filepath):
        with open(split_filepath, 'r') as f:
            return json.load(f)

    # If no split exists, create one based on available parent folders
    parent_video_ids = [d for d in os.listdir(clips_dir) if os.path.isdir(os.path.join(clips_dir, d))]
    random.seed(42) # Reproducibility
    random.shuffle(parent_video_ids)

    total = len(parent_video_ids)
    train_end = int(total * 0.8) # ~80%
    val_end = train_end + int(total * 0.1) # ~10%

    splits = {
        "train": parent_video_ids[:train_end],
        "val": parent_video_ids[train_end:val_end],
        "test": parent_video_ids[val_end:]
    }

    # Save to disk
    with open(split_filepath, 'w') as f:
        json.dump(splits, f, indent=4)
    
    print(f"Created new dataset splits: {len(splits['train'])} Train, {len(splits['val'])} Val, {len(splits['test'])} Test parent videos.")
    return splits

def get_split_for_clip(parent_video_id, splits):
    for split_name, ids in splits.items():
        if parent_video_id in ids:
            return split_name
    return "train" # Default fallback

# ==========================================
# 4. Main Pipeline Logic
# ==========================================
def run_dataset_formatter(clips_dir, output_dir, model_name, base_url, api_key, log_filename, split_filename, max_retries):
    print("\n=== DATASET FORMATTER (SFT & GRPO) ===\n")
    
    os.makedirs(output_dir, exist_ok=True)
    log_file_path = os.path.join(output_dir, log_filename)
    setup_log_csv(log_file_path)

    # 1. Manage Splits & Resuming
    splits = manage_train_val_test_split(clips_dir, output_dir, split_filename)
    processed_clips = get_processed_clips(log_file_path)
    print(f"ℹ️ Resuming... {len(processed_clips)} clips already formatted.")

    # 2. Setup LLM Client (Ollama via OpenAI SDK)
    client = OpenAI(base_url=base_url, api_key=api_key)

    # 3. Gather all JSONL clip metadata files
    all_clips = []
    for parent_id in os.listdir(clips_dir):
        parent_dir = os.path.join(clips_dir, parent_id)
        if not os.path.isdir(parent_dir): continue
        
        for file in os.listdir(parent_dir):
            if file.endswith('.jsonl'):
                all_clips.append(os.path.join(parent_dir, file))

    if not all_clips:
        print("⚠️ No clips found in the clip directory.")
        return

    # 4. Open Output Files (Append Mode)
    file_handles = {}
    for split_type in ["train", "val", "test"]:
        for format_type in ["sft", "grpo"]:
            file_path = os.path.join(output_dir, f"{split_type}_{format_type}.jsonl")
            file_handles[f"{split_type}_{format_type}"] = open(file_path, 'a', encoding='utf-8')

    # 5. Process Clips
    for clip_path in tqdm(all_clips, desc="Generating VLM Formats"):
        # Read clip metadata
        with open(clip_path, 'r', encoding='utf-8') as f:
            try:
                clip_data = json.loads(f.readline())
            except json.JSONDecodeError:
                continue
        
        clip_id = clip_data.get("clip_id")
        parent_video_id = clip_data.get("parent_video_id")
        clip_filename = clip_data.get("clip_filename")
        
        if not clip_id or clip_id in processed_clips:
            continue
            
        split_group = get_split_for_clip(parent_video_id, splits)
        video_path = f"{parent_video_id}/{clip_filename}" # Relative path to clip

        # --- A. Construct LLM Input Context ---
        user_prompt = f"""You are watching a cataract surgery video clip (ID: {clip_id}).

Based on this surgical context, generate exactly 3 multiple-choice questions:

Surgical Step: {clip_data.get("step_title", "")}

What's happening in the video:
{clip_data.get("visual_description", "")}

Instruments visible: {", ".join(clip_data.get("instruments", []))}
Anatomical structures: {", ".join(clip_data.get("anatomy", []))}

Generate:
1. One "step_identification" question about what surgical step is being performed
2. One "visual_observation" question about specific visual details or techniques shown
3. One "instrument_identification" question about the surgical tools being used

IMPORTANT: Write your reasoning naturally, as if describing what you observe in the video. Do NOT reference metadata field names."""

        # --- B. Call LLM & Enforce Schema ---
        llm_success = False
        generated_qa_pairs = []
        
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.7
                )
                
                raw_output = response.choices[0].message.content
                parsed_json = json.loads(raw_output)
                
                # Schema Check using Pydantic
                validated_data = QAList(**parsed_json)
                generated_qa_pairs = validated_data.qa_pairs
                llm_success = True
                break # Success, exit retry loop
                
            except (json.JSONDecodeError, ValidationError) as e:
                tqdm.write(f"⚠️ Validation failed on attempt {attempt+1} for {clip_id}: {e}")
            except Exception as e:
                tqdm.write(f"❌ LLM Error on attempt {attempt+1} for {clip_id}: {e}")
                time.sleep(2)

        if not llm_success:
            log_progress(log_file_path, clip_id, "ERROR_LLM")
            continue

        # --- C. Format and Save Data ---
        sft_handle = file_handles[f"{split_group}_sft"]
        grpo_handle = file_handles[f"{split_group}_grpo"]

        # Task 1: SFT Description Task (Deterministic based on existing description)
        sft_description = {
            "messages": [
                {"role": "user", "content": [
                    {"type": "video", "video": video_path},
                    {"type": "text", "text": "Describe what is happening in this cataract surgical video clip."}
                ]},
                {"role": "assistant", "content": clip_data.get("visual_description", "")}
            ]
        }
        sft_handle.write(json.dumps(sft_description) + "\n")

        # Task 2: Process LLM Generated QA Pairs into SFT and GRPO
        for qa in generated_qa_pairs:
            # Construct the multiple choice text block
            options_text = "\n".join([f"{k}) {v}" for k, v in qa.options.items()])
            question_text = f"{qa.question}\n{options_text}\n\nProvide your reasoning first, then state your answer."
            assistant_answer = f"{qa.reference_reasoning} Therefore the answer is {qa.correct_answer}) {qa.options[qa.correct_answer]}."

            # SFT Format
            sft_qa = {
                "messages": [
                    {"role": "user", "content": [
                        {"type": "video", "video": video_path},
                        {"type": "text", "text": question_text}
                    ]},
                    {"role": "assistant", "content": assistant_answer}
                ]
            }
            sft_handle.write(json.dumps(sft_qa) + "\n")

            # GRPO Format
            reward_type = "deterministic" if qa.question_type in ["step_identification", "instrument_identification"] else "llm_judge"
            grpo_qa = {
                "prompt": [
                    {"role": "user", "content": [
                        {"type": "video", "video": video_path},
                        {"type": "text", "text": question_text}
                    ]}
                ],
                "correct_answer": qa.correct_answer,
                "question_type": qa.question_type,
                "reference_reasoning": qa.reference_reasoning,
                "reward_type": reward_type
            }
            grpo_handle.write(json.dumps(grpo_qa) + "\n")

        # Mark clip as successful
        log_progress(log_file_path, clip_id, "SUCCESS")

    # 6. Cleanup
    for handle in file_handles.values():
        handle.close()

    print("\n" + "="*30)
    print("   DATASET FORMATTING COMPLETED")
    print("="*30)
    print(f"📂 Output Folder: {os.path.abspath(output_dir)}")
    print(f"📊 Splits used: {len(splits['train'])} Train, {len(splits['val'])} Val, {len(splits['test'])} Test parent videos.")

