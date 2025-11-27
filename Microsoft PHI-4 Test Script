# ============================================================================
# PHI-4-MULTIMODAL VIDEO EVENT TEST SCRIPT
# Model: microsoft/Phi-4-multimodal-instruct
# ============================================================================
"""
Tests Phi-4-multimodal-instruct on ONE video to verify it works.
Run this before moving to HPC.
"""

# ============================================================================
# CONFIGURATION
# ============================================================================

BATCH_NUMBER = 1
DRIVE_BASE = "/content/drive/MyDrive/SoccerNet_EPL"
FPS = 1.0
FRAMES_PER_CHUNK = 2  # still used for chunking, but we only test first 10 frames

# ============================================================================
# INSTALLATION
# ============================================================================

import subprocess
import sys

print("📦 Installing packages...")

# Uninstall current transformers version to ensure a clean install of the latest
subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', 'transformers'], check=False)
# Install the latest stable transformers version
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'transformers'], check=False)

packages = ["accelerate", "torch", "pillow", "torchvision"]
for package in packages:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", package],
        check=False,
    )

print("✅ Packages installed")

# ============================================================================
# IMPORTS
# ============================================================================

from google.colab import drive
import torch
import os
import json
import subprocess
import shutil
from pathlib import Path
from datetime import datetime
from PIL import Image
import gc

# ============================================================================
# HUGGING FACE AUTH
# ============================================================================

from huggingface_hub import login
from google.colab import userdata

try:
    HF_TOKEN = userdata.get("HF_TOKEN")
    if HF_TOKEN is not None:
        login(token=HF_TOKEN)
        print("✅ Hugging Face login successful")
    else:
        print("⚠️  HF_TOKEN not set in Colab userdata, proceeding without explicit login")
except Exception as e:
    print(f"❌ HF login failed: {e}")

if torch.cuda.is_available():
    print(f"\n🖥️  GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
    )
else:
    print("\n⚠️  No CUDA GPU detected. This script expects a GPU runtime.")

# ============================================================================
# MOUNT DRIVE & FIND VIDEO
# ============================================================================

print("\n📁 Mounting Google Drive...")
drive.mount("/content/drive")

batch_dir = os.path.join(DRIVE_BASE, f"batch_{BATCH_NUMBER}")
video_files = []
for root, dirs, files in os.walk(batch_dir):
    for file in files:
        if file.endswith(".mkv"):
            video_files.append(os.path.join(root, file))

if not video_files:
    raise FileNotFoundError(f"No .mkv files found in {batch_dir}")

print(f"✅ Found {len(video_files)} videos")
print(f"📹 Testing with: {Path(video_files[0]).name}")

TEST_VIDEO = video_files[0]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def extract_frames(video_path, fps=1.0):
    """Extract frames from video via ffmpeg."""
    output_dir = Path("/content/temp_frames") / Path(video_path).stem
    output_dir.mkdir(parents=True, exist_ok=True)
    output_pattern = str(output_dir / "frame_%04d.jpg")

    cmd = [
        "ffmpeg",
        "-i",
        video_path,
        "-vf",
        f"fps={fps}",
        "-q:v",
        "2",
        "-y",
        output_pattern,
    ]

    print(f"   Running: {' '.join(cmd[:5])}...")
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if result.returncode != 0:
        print(f"❌ ffmpeg failed with return code {result.returncode}")
        print("   Error output:")
        print(result.stderr.decode())
        return []

    frames = sorted(output_dir.glob("frame_*.jpg"))
    print(f"📸 Extracted {len(frames)} frames")
    return [str(f) for f in frames]


def load_images(frame_paths):
    """Load images from disk as RGB PIL images."""
    images = []
    for path in frame_paths:
        try:
            images.append(Image.open(path).convert("RGB"))
        except Exception as e:
            print(f"⚠️  Error loading {path}: {e}")
    return images


def chunk_frames(frames, frames_per_chunk=45):
    """Split frame paths into chunks."""
    chunks = []
    for i in range(0, len(frames), frames_per_chunk):
        chunks.append(frames[i : i + frames_per_chunk])
    return chunks


# ============================================================================
# LOAD PHI-4-MULTIMODAL MODEL
# ============================================================================

print("\n" + "=" * 80)
print("🤖 LOADING Phi-4-multimodal-instruct")
print("=" * 80)

from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    GenerationConfig,
)

MODEL_ID = "microsoft/Phi-4-multimodal-instruct"

# Load processor and model (vision + text)
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

# Use eager attention to avoid needing flash-attn on older GPUs
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="cuda" if torch.cuda.is_available() else "cpu",
    torch_dtype="auto",
    trust_remote_code=True,
    _attn_implementation="eager",
)

generation_config = GenerationConfig.from_pretrained(MODEL_ID)

if torch.cuda.is_available():
    print(
        f"✅ Phi-4-multimodal loaded (VRAM: {torch.cuda.memory_allocated()/1024**3:.1f} GB)"
    )
else:
    print("✅ Phi-4-multimodal loaded on CPU (expect this to be VERY slow)")

# ============================================================================
# TEST ANNOTATION
# ============================================================================

print("\n" + "=" * 80)
print("🎬 TESTING VIDEO ANNOTATION (PHI-4-MULTIMODAL)")
print("=" * 80)

# Extract frames
print("\n📸 Extracting frames...")
print(f"   Video path: {TEST_VIDEO}")
print(f"   Video exists: {os.path.exists(TEST_VIDEO)}")
print(f"   Video size: {os.path.getsize(TEST_VIDEO) / (1024**3):.2f} GB")

frame_paths = extract_frames(TEST_VIDEO, fps=FPS)

if not frame_paths:
    print("❌ Frame extraction failed - check video path or ffmpeg")
    raise SystemExit(1)

# Chunk frames then only use first chunk's first 10 frames for this test
chunks = chunk_frames(frame_paths, FRAMES_PER_CHUNK)
print(f"📦 Total chunks: {len(chunks)}")

if len(chunks) == 0:
    print("❌ No chunks created - no frames extracted")
    raise SystemExit(1)

test_frames = chunks[0][:10]  # First 10 frames only for memory efficiency
print(f"🧪 Testing with FIRST {len(test_frames)} FRAMES")

images = load_images(test_frames)
if not images:
    print("❌ No images successfully loaded")
    raise SystemExit(1)

print(f"\n🤖 Running Phi-4 inference on {len(images)} images...")

# ============================================================================
# BUILD PHI-4 CHAT / VISION PROMPT
# ============================================================================

# Phi-4 chat tokens
USER_TOK = "<|user|>"
ASSISTANT_TOK = "<|assistant|>"
END_TOK = "<|end|>"

# Create image placeholders: <|image_1|><|image_2|>... for each frame
image_placeholders = "".join(f"<|image_{i+1}|>" for i in range(len(images)))

task_prompt = f"""You must respond with ONLY valid JSON. No markdown, no explanations, no formatting.

Analyze these {len(images)} soccer frames (0 to {len(images)-1}).

Detect events: Goals, Shots, Fouls, Cards, Corners, Penalties.

CRITICAL: Your ENTIRE response must be ONLY this JSON structure with NO additional text:

{{"chunk": 1, "start_frame": 0, "end_frame": {len(images)-1}, "events": [{{"event_type": "shot", "frame_number": 3, "timestamp": "00:03", "team": "home", "player": "#10", "confidence": 0.9, "description": "shot on goal"}}, {{"event_type": "corner", "frame_number": 5, "timestamp": "00:05", "team": "away", "player": "#7", "confidence": 0.8, "description": "corner kick"}}]}}

Do NOT use markdown. Do NOT add explanations. Output ONLY the JSON object."""

# Phi-4 vision format (multiple images):
# <|user|><|image_1|><|image_2|>...{text}<|end|><|assistant|>
full_prompt = f"{USER_TOK}{image_placeholders}{task_prompt}{END_TOK}{ASSISTANT_TOK}"

inputs = processor(
    text=full_prompt,
    images=images,
    return_tensors="pt",
)

device = "cuda" if torch.cuda.is_available() else "cpu"
inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

print("⏳ Generating response...")
start_time = datetime.now()

with torch.no_grad():
    generate_ids = model.generate(
        **inputs,
        max_new_tokens=4096,  # generous to allow full JSON
        generation_config=generation_config,
    )

# Strip off the prompt tokens and decode only new text
new_tokens = generate_ids[:, inputs["input_ids"].shape[1]:]
output_text = processor.batch_decode(
    new_tokens,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)[0]

elapsed = (datetime.now() - start_time).total_seconds()
print(f"✅ Generation complete ({elapsed:.1f}s)")

# ============================================================================
# PARSE RESULT AS JSON
# ============================================================================

print("\n" + "=" * 80)
print("📊 RESULTS")
print("=" * 80)

print("\n📄 Raw output (first 1000 chars):")
print(output_text[:1000])
print("\n" + "-" * 80)

# Some safety trimming in case it echoes parts of the prompt
if "Analyze these" in output_text:
    json_start_hint = output_text.find('{"chunk"}')
    if json_start_hint != -1:
        output_text = output_text[json_start_hint:]

try:
    # Look for JSON object starting at "{"chunk""
    json_start = output_text.find('{"chunk"}')

    if json_start != -1:
        json_text = output_text[json_start:]

        # If JSON seems truncated, try a minimal repair
        if not json_text.rstrip().endswith("}"):
            print("\n⚠️  JSON appears truncated, attempting repair...")
            if '"events":' in json_text:
                last_brace = json_text.rfind("}")
                if last_brace != -1:
                    json_text = json_text[: last_brace + 1]
                    # In worst case, user can fix manually later

        result = json.loads(json_text)

        print("\n✅ JSON parsed successfully!")
        print(f"   Events detected: {len(result.get('events', []))}")

        if result.get("events"):
            print("\n📋 Events found:")
            for event in result["events"]:
                print(
                    f"   • {event.get('event_type', 'unknown')} "
                    f"at frame {event.get('frame_number', 'N/A')}"
                )

        print("\n📄 Full JSON:")
        print(json.dumps(result, indent=2))
    else:
        raise json.JSONDecodeError("No JSON object found", output_text, 0)

except (json.JSONDecodeError, ValueError) as e:
    print(f"\n⚠️  Direct JSON extraction failed: {e}")
    print("\n🔄 Attempting to parse loose JSON objects via regex...")

    import re

    events = []

    # Find all JSON object-like snippets in the text
    json_objects = re.findall(r"\{[^}]+\}", output_text)

    for obj_str in json_objects:
        try:
            event = json.loads(obj_str)
            events.append(event)
        except Exception:
            continue

    if events:
        result = {
            "chunk": 1,
            "start_frame": 0,
            "end_frame": len(images) - 1,
            "events": events,
        }

        print(f"✅ Parsed {len(events)} events from loose JSON snippets")
        print("\n📋 Sample events:")
        for event in events[:3]:
            print(
                f"   • {event.get('event_type', 'unknown')} "
                f"at frame {event.get('frame_number', 'N/A')}"
            )

        print("\n📄 Reconstructed JSON:")
        print(json.dumps(result, indent=2))
    else:
        print("❌ Could not extract any valid JSON objects")
        result = {"error": "No valid JSON found", "events": []}

# ============================================================================
# CLEANUP
# ============================================================================

print("\n🧹 Cleaning up...")
if frame_paths:
    shutil.rmtree(Path(frame_paths[0]).parent, ignore_errors=True)
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("\n" + "=" * 80)
print("✅ PHI-4-MULTIMODAL TEST COMPLETE")
print("=" * 80)
print(f"⏱️  Time for {len(images)} frames: {elapsed:.1f}s")
if torch.cuda.is_available():
    print(f"💾 VRAM used: {torch.cuda.memory_allocated()/1024**3:.1f} GB")
