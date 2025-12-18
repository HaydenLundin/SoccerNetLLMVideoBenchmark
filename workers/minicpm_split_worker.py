import os
import sys
import argparse

# Parse args FIRST to get GPU ID before importing torch
parser = argparse.ArgumentParser()
parser.add_argument("--gpu_id", type=int, required=True)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--video_index", type=int, default=0, help="Index of video to process")
args = parser.parse_args()

# CRITICAL: Set CUDA_VISIBLE_DEVICES BEFORE importing torch
# This isolates each worker to only see its assigned GPU
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

import json
import subprocess
import shutil
import torch
from pathlib import Path
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from huggingface_hub import login

# ============================================================================
# CONFIGURATION - TITAN RTX OPTIMIZED FOR MINICPM-V 2.6
# ============================================================================

BASE_DIR = os.path.join(os.environ['HOME'], "soccer_project")
VIDEO_DIR = os.path.join(BASE_DIR, "raw_data")
TEMP_FRAME_DIR = os.path.join(BASE_DIR, "temp_frames")

# QWEN-MATCHED SETTINGS (For fair comparison)
FPS = 5.0              # Match Qwen's 5 FPS
WINDOW_SECONDS = 3.0   # Match Qwen's 3s window = 15 frames
STEP_SECONDS = 2.0     # Match Qwen's 2s step (50% overlap)
TARGET_DURATION = 3000  # Process first 50 mins (full match half)

# ============================================================================
# WORKER SETUP
# ============================================================================

# Calculate Time Slice for this GPU
slice_duration = TARGET_DURATION / args.num_workers
my_start_time = args.gpu_id * slice_duration
my_end_time = my_start_time + slice_duration

# After CUDA_VISIBLE_DEVICES is set, the only visible GPU is cuda:0
device_id = "cuda:0"

print(f"👷 [GPU {args.gpu_id}] Timeline: {my_start_time:.1f}s to {my_end_time:.1f}s")

output_filename = os.path.join(BASE_DIR, f"partial_results_minicpm_vid{args.video_index}_gpu{args.gpu_id}.json")

# ============================================================================
# MODEL SETUP (MiniCPM-V 2.6 INT4 - ~7GB VRAM)
# ============================================================================

HF_TOKEN = os.getenv('HF_TOKEN')
if HF_TOKEN:
    try:
        login(token=HF_TOKEN)
    except Exception as e:
        print(f"⚠️ [GPU {args.gpu_id}] HF login failed (rate limit?): {e}")
        print(f"   Continuing anyway - model will use cached files or public access")

print(f"🤖 [GPU {args.gpu_id}] Loading MiniCPM-V 2.6 INT4 ({int(FPS*WINDOW_SECONDS)} frames/window)...")

# Load the INT4 quantized model (~7GB VRAM)
# Use device_map to assign to specific GPU
model = AutoModel.from_pretrained(
    'openbmb/MiniCPM-V-2_6-int4',
    trust_remote_code=True,
    device_map=device_id
)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(
    'openbmb/MiniCPM-V-2_6-int4',
    trust_remote_code=True
)

print(f"✅ [GPU {args.gpu_id}] Model loaded successfully")

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
    # Get team context from first frame
    frames = extract_clip_frames(video_path, start_time=15, duration=0.1, fps=1)
    if not frames: return "Home (Left), Away (Right)"

    image = Image.open(frames[0]).convert('RGB')

    # MiniCPM-V format: image + question in content list
    msgs = [
        {
            "role": "user",
            "content": [image, "Identify the HOME team (left) and AWAY team (right) names and colors from the scoreboard."]
        }
    ]

    try:
        with torch.no_grad():
            result = model.chat(
                image=None,
                msgs=msgs,
                tokenizer=tokenizer,
                max_new_tokens=128
            )
    except Exception as e:
        print(f"⚠️ Failed to get match context: {e}")
        result = "Home (Left), Away (Right)"
    finally:
        shutil.rmtree(Path(frames[0]).parent, ignore_errors=True)
        torch.cuda.empty_cache()

    return result

# Find Video
video_files = []
for root, dirs, files in os.walk(VIDEO_DIR):
    for file in files:
        if file.endswith('.mkv'):
            video_files.append(os.path.join(root, file))
if args.video_index < len(video_files):
    TEST_VIDEO = video_files[args.video_index]
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

    # Load all frames (same as Qwen: 15 frames at 5 FPS)
    images = [Image.open(f).convert('RGB') for f in frames]

    # Build prompt - STRICT JSON OUTPUT FORMAT
    prompt = (
        f"Context: {match_context}\n\n"
        f"Analyze these {len(images)} frames from a {WINDOW_SECONDS}s soccer clip.\n\n"
        "TASK: Detect soccer events from this list ONLY:\n"
        "Goal, Shot on target, Shot off target, Foul, Yellow card, Red card, Offside, "
        "Corner, Free-kick, Penalty, Throw-in, Kick-off, Goal kick, Substitution, Ball out of play, Clearance\n\n"
        "OUTPUT FORMAT: You MUST output ONLY valid JSON. No explanations, no markdown, no bullet points.\n"
        "For each event, output exactly this format on one line:\n"
        '{"label": "EVENT_NAME", "team": "home", "confidence": 0.85, "details": "brief description"}\n\n'
        "RULES:\n"
        '- "team" MUST be exactly "home" or "away" (not team names)\n'
        '- "label" MUST be one of the 16 events listed above\n'
        "- Output one JSON object per line\n"
        "- If no events detected, output exactly: None\n\n"
        "OUTPUT:"
    )

    # MiniCPM-V format: images + question in content list
    msgs = [
        {
            "role": "user",
            "content": images + [prompt]
        }
    ]

    try:
        with torch.no_grad():
            result = model.chat(
                image=None,
                msgs=msgs,
                tokenizer=tokenizer,
                max_new_tokens=256
            )
    except Exception as e:
        print(f"⚠️ [GPU {args.gpu_id}] Error at {current_time:.1f}s: {e}")
        result = "None"

    if "None" not in result and result.strip():
        print(f"⚡ [GPU {args.gpu_id}] {current_time:.1f}s: {result}")
        events_found.append({"time": current_time, "raw": result})

        # Incremental Save
        try:
            with open(output_filename, "w") as f:
                json.dump(events_found, f, indent=2)
        except Exception as e:
            print(f"⚠️ Save failed: {e}")

    # Clean up
    del images
    torch.cuda.empty_cache()
    shutil.rmtree(Path(frames[0]).parent, ignore_errors=True)

    current_time += STEP_SECONDS

print(f"✅ [GPU {args.gpu_id}] Finished.")
