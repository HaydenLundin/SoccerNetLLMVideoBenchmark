import os
import sys
import json
import subprocess
import shutil
import torch
import argparse
from pathlib import Path
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from huggingface_hub import login

# ============================================================================
# CONFIGURATION - TITAN RTX SAFE MODE
# ============================================================================

BASE_DIR = os.path.join(os.environ['HOME'], "soccer_project")
VIDEO_DIR = os.path.join(BASE_DIR, "raw_data")
TEMP_FRAME_DIR = os.path.join(BASE_DIR, "temp_frames")

# PHYSICS OPTIMIZED SETTINGS (Fits in ~11GB VRAM)
FPS = 5.0             # Keep 10 FPS for ball physics
WINDOW_SECONDS = 3.0   # 2s Window (Critical for Turing memory limit)
STEP_SECONDS = 2.0     # 50% Overlap to catch split events
TARGET_DURATION = 3000  # Process first 5 mins (Set to 2700 for full half)

# ============================================================================
# WORKER SETUP
# ============================================================================

parser = argparse.ArgumentParser()
parser.add_argument("--gpu_id", type=int, required=True)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--video_index", type=int, default=0, help="Index of video to process")
args = parser.parse_args()

# Calculate Time Slice for this GPU
slice_duration = TARGET_DURATION / args.num_workers
my_start_time = args.gpu_id * slice_duration
my_end_time = my_start_time + slice_duration

device_id = f"cuda:{args.gpu_id}"
print(f"👷 [GPU {args.gpu_id}] Timeline: {my_start_time:.1f}s to {my_end_time:.1f}s")

output_filename = os.path.join(BASE_DIR, f"partial_results_vid_qwen7b_vid{args.video_index}_gpu{args.gpu_id}.json")
# ============================================================================
# MODEL SETUP (7B + 4-BIT QUANTIZATION)
# ============================================================================

HF_TOKEN = os.getenv('HF_TOKEN')
if HF_TOKEN: login(token=HF_TOKEN)

print(f"🤖 [GPU {args.gpu_id}] Loading Qwen 7B (BnB 4-bit)...")

# 1. Define 4-bit Configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# 2. Load the STANDARD model
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",  # Standard Model
    quantization_config=bnb_config, # Force 4-bit
    device_map=device_id            # Isolate to this GPU
)

processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    min_pixels=256*28*28,
    max_pixels=480*28*28            # 480p Resolution
)

# ============================================================================
# PROCESSING
# ============================================================================

def extract_clip_frames(video_path, start_time, duration, fps):
    output_dir = Path(TEMP_FRAME_DIR) / f"worker_{args.gpu_id}" / f"clip_{int(start_time)}"
    if output_dir.exists(): shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        'ffmpeg', '-ss', str(start_time), '-t', str(duration),
        '-i', video_path, '-vf', f'fps={fps}', '-q:v', '2', '-y',
        str(output_dir / "frame_%04d.jpg")
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return sorted([str(f) for f in output_dir.glob("frame_*.jpg")])

def get_match_context(video_path):
    # Only run calibration once (everyone gets same context)
    frames = extract_clip_frames(video_path, start_time=15, duration=0.1, fps=1)
    if not frames: return "Home (Left), Away (Right)"

    image = Image.open(frames[0]).convert('RGB')
    prompt = "Identify HOME team (Left) and AWAY team (Right) names and colors from the scoreboard."

    conversation = [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image", "image": image}]}]
    text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(text=[text_prompt], images=[image], return_tensors="pt")
    inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=128)

    # FIX: Use brackets for dictionary access
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs['input_ids'], output_ids)]
    result = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    shutil.rmtree(Path(frames[0]).parent, ignore_errors=True)
    return result

# Find Video
video_files = []
for root, dirs, files in os.walk(VIDEO_DIR):
    for file in files:
        if file.endswith('.mkv'):
            video_files.append(os.path.join(root, file))
if args.video_index < len(video_files):
    TEST_VIDEO = video_files[args.video_index]
    # print(f"📹 Processing Video {args.video_index}: {os.path.basename(TEST_VIDEO)}")
else:
    print(f"❌ Index {args.video_index} out of range.")
    sys.exit()

# Get Context
match_context = get_match_context(TEST_VIDEO)
current_time = my_start_time
events_found = []

# Main Loop
while current_time < my_end_time:
    frames = extract_clip_frames(TEST_VIDEO, current_time, WINDOW_SECONDS, FPS)
    if not frames: break

    images = [Image.open(f).convert('RGB') for f in frames]

    # --- THE RESTORED ADVANCED PROMPT ---
    prompt = (
        f"Context: {match_context}\n"
        f"Analyze this {WINDOW_SECONDS}s clip. Detect ANY of these 17 events:\n"
        "- Goals/Shots: Goal, Shot on target, Shot off target\n"
        "- Fouls/Cards: Foul, Yellow card, Red card, Offside\n"
        "- Set Pieces: Corner, Free-kick, Penalty, Throw-in, Kick-off, Goal kick\n"
        "- Other: Substitution, Ball out of play, Clearance\n\n"
        "For EACH event, output a JSON object:\n"
        "{'label': 'EVENT_TYPE', 'team': 'home' OR 'away', 'confidence': 0.0-1.0, 'details': 'DESC'}\n"
        "If nothing significant happens, output exactly: None."
    )

    conversation = [{"role": "user", "content": [{"type": "text", "text": prompt}] + [{"type": "image"} for _ in images]}]
    text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(text=[text_prompt], images=images, return_tensors="pt")
    inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=256)

    # FIX: Use brackets for dictionary access
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs['input_ids'], output_ids)]
    result = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    if "None" not in result:
        print(f"⚡ [GPU {args.gpu_id}] {current_time:.1f}s: {result}")
        events_found.append({"time": current_time, "raw": result})

        # Incremental Save (Overwrites file with latest list every time)
        try:
            with open(output_filename, "w") as f:
                json.dump(events_found, f, indent=2)
        except Exception as e:
            print(f"⚠️ Save failed: {e}")

    del inputs, images, output_ids
    torch.cuda.empty_cache()
    shutil.rmtree(Path(frames[0]).parent, ignore_errors=True)

    current_time += STEP_SECONDS

print(f"✅ [GPU {args.gpu_id}] Finished.")
