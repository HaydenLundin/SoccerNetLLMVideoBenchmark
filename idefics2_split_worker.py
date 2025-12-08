import os
import sys
import json
import subprocess
import shutil
import torch
import argparse
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, Idefics2ForConditionalGeneration, BitsAndBytesConfig
from huggingface_hub import login

# ============================================================================
# CONFIGURATION - TITAN RTX OPTIMIZED
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

output_filename = os.path.join(BASE_DIR, f"partial_results_idefics2_vid{args.video_index}_gpu{args.gpu_id}.json")

# ============================================================================
# MODEL SETUP (Idefics2-8B + 4-BIT QUANTIZATION)
# ============================================================================

HF_TOKEN = os.getenv('HF_TOKEN')
if HF_TOKEN: login(token=HF_TOKEN)

print(f"🤖 [GPU {args.gpu_id}] Loading Idefics2-8B (BnB 4-bit, {int(FPS*WINDOW_SECONDS)} frames/window)...")

# 1. Define 4-bit Configuration (same as Qwen)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# 2. Load Idefics2-8B model (optimized for multi-image tasks)
model = Idefics2ForConditionalGeneration.from_pretrained(
    "HuggingFaceM4/idefics2-8b",
    quantization_config=bnb_config,
    device_map=device_id,
    torch_dtype=torch.float16
)

processor = AutoProcessor.from_pretrained(
    "HuggingFaceM4/idefics2-8b"
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
    # Get team context from first frame
    frames = extract_clip_frames(video_path, start_time=15, duration=0.1, fps=1)
    if not frames: return "Home (Left), Away (Right)"

    image = Image.open(frames[0]).convert('RGB')

    # Idefics2 format: text with <image> placeholder
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Identify the HOME team (left) and AWAY team (right) names and colors from the scoreboard."},
            ]
        }
    ]

    try:
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=prompt, images=[image], return_tensors="pt")
        inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=128)

        result = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        # Extract after assistant marker
        if "Assistant:" in result:
            result = result.split("Assistant:")[-1].strip()

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

    # Idefics2 format: multiple <image> tokens
    image_tokens = " ".join(["<image>"] * len(images))
    text = f"{image_tokens}\n\nContext: {match_context}\n\nAnalyze these {len(images)} frames from a {WINDOW_SECONDS}s soccer clip. Detect ANY of these 17 soccer events:\n- Goals/Shots: Goal, Shot on target, Shot off target\n- Fouls/Cards: Foul, Yellow card, Red card, Offside\n- Set Pieces: Corner, Free-kick, Penalty, Throw-in, Kick-off, Goal kick\n- Other: Substitution, Ball out of play, Clearance\n\nFor EACH event detected, output a JSON object:\n{{'label': 'EVENT_TYPE', 'team': 'home' OR 'away', 'confidence': 0.0-1.0, 'details': 'brief description'}}\nIf nothing significant happens, output: None"

    messages = [
        {
            "role": "user",
            "content": [
                *[{"type": "image"} for _ in images],
                {"type": "text", "text": text},
            ]
        }
    ]

    # Process - OOM will fail the job
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=images, return_tensors="pt")
    inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)

    result = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    # Extract after assistant marker
    if "Assistant:" in result:
        result = result.split("Assistant:")[-1].strip()

    if "None" not in result and result:
        print(f"⚡ [GPU {args.gpu_id}] {current_time:.1f}s: {result}")
        events_found.append({"time": current_time, "raw": result})

        # Incremental Save
        try:
            with open(output_filename, "w") as f:
                json.dump(events_found, f, indent=2)
        except Exception as e:
            print(f"⚠️ Save failed: {e}")

    # Clean up
    del inputs, output_ids, images
    torch.cuda.empty_cache()
    shutil.rmtree(Path(frames[0]).parent, ignore_errors=True)

    current_time += STEP_SECONDS

print(f"✅ [GPU {args.gpu_id}] Finished.")
