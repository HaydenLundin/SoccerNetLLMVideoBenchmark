# ============================================================================
# LLAMA-3.2-11B-VISION TEST SCRIPT
# ============================================================================
"""
Tests Llama-3.2-11B-Vision on ONE video to verify it works.
Run this before moving to HPC.
"""

# ============================================================================
# CONFIGURATION
# ============================================================================

BATCH_NUMBER = 1
DRIVE_BASE = "/content/drive/MyDrive/SoccerNet_EPL"
FPS = 1.0
FRAMES_PER_CHUNK = 45

# ============================================================================
# INSTALLATION
# ============================================================================

import subprocess
import sys

print("📦 Installing packages...")
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                'git+https://github.com/huggingface/transformers'], check=False)

packages = ['accelerate', 'torch', 'pillow', 'torchvision']
for package in packages:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', package], check=False)

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
    HF_TOKEN = userdata.get('HF_TOKEN')
    login(token=HF_TOKEN)
    print("✅ Hugging Face login successful")
except Exception as e:
    print(f"❌ HF login failed: {e}")

print(f"\n🖥️  GPU: {torch.cuda.get_device_name(0)}")
print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# ============================================================================
# MOUNT DRIVE & FIND VIDEO
# ============================================================================

print("\n📁 Mounting Google Drive...")
drive.mount('/content/drive')

batch_dir = os.path.join(DRIVE_BASE, f"batch_{BATCH_NUMBER}")
video_files = []
for root, dirs, files in os.walk(batch_dir):
    for file in files:
        if file.endswith('.mkv'):
            video_files.append(os.path.join(root, file))

print(f"✅ Found {len(video_files)} videos")
print(f"📹 Testing with: {Path(video_files[0]).name}")

TEST_VIDEO = video_files[0]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_frames(video_path, fps=1.0):
    """Extract frames from video"""
    output_dir = Path("/content/temp_frames") / Path(video_path).stem
    output_dir.mkdir(parents=True, exist_ok=True)
    output_pattern = str(output_dir / "frame_%04d.jpg")

    cmd = ['ffmpeg', '-i', video_path, '-vf', f'fps={fps}', '-q:v', '2', '-y', output_pattern]

    print(f"   Running: {' '.join(cmd[:5])}...")
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if result.returncode != 0:
        print(f"❌ ffmpeg failed with return code {result.returncode}")
        print(f"   Error output:")
        print(result.stderr.decode())
        return []

    frames = sorted(output_dir.glob("frame_*.jpg"))
    print(f"📸 Extracted {len(frames)} frames")
    return [str(f) for f in frames]

def load_images(frame_paths):
    """Load images from disk"""
    images = []
    for path in frame_paths:
        try:
            images.append(Image.open(path).convert('RGB'))
        except Exception as e:
            print(f"⚠️  Error loading {path}: {e}")
    return images

def chunk_frames(frames, frames_per_chunk=45):
    """Split frames into chunks"""
    chunks = []
    for i in range(0, len(frames), frames_per_chunk):
        chunks.append(frames[i:i + frames_per_chunk])
    return chunks

# ============================================================================
# LOAD LLAMA MODEL
# ============================================================================

print("\n" + "="*80)
print("🤖 LOADING LLAMA-3.2-11B-VISION")
print("="*80)

from transformers import MllamaForConditionalGeneration, AutoProcessor

model = MllamaForConditionalGeneration.from_pretrained(
    "meta-llama/Llama-3.2-11B-Vision-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("meta-llama/Llama-3.2-11B-Vision-Instruct")

print(f"✅ Llama loaded (VRAM: {torch.cuda.memory_allocated()/1024**3:.1f} GB)")

# ============================================================================
# TEST ANNOTATION
# ============================================================================

print("\n" + "="*80)
print("🎬 TESTING VIDEO ANNOTATION")
print("="*80)

# Extract frames
print("\n📸 Extracting frames...")
print(f"   Video path: {TEST_VIDEO}")
print(f"   Video exists: {os.path.exists(TEST_VIDEO)}")
print(f"   Video size: {os.path.getsize(TEST_VIDEO) / (1024**3):.2f} GB")

frame_paths = extract_frames(TEST_VIDEO, fps=FPS)

if not frame_paths:
    print("❌ Frame extraction failed - check video path or ffmpeg")
    exit()

# Take only first 10 images for testing (45 is too much!)
chunks = chunk_frames(frame_paths, FRAMES_PER_CHUNK)
print(f"📦 Total chunks: {len(chunks)}")

if len(chunks) == 0:
    print("❌ No chunks created - no frames extracted")
    exit()

# Use only 10 images for memory efficiency
test_frames = chunks[0][:10]  # First 10 frames only
print(f"🧪 Testing with FIRST 10 FRAMES (reduced for memory)")

test_chunk = test_frames
images = load_images(test_chunk)

print(f"\n🤖 Running Llama inference on {len(images)} images...")

# Build prompt - ULTRA STRICT JSON FORMAT
prompt = f"""You must respond with ONLY valid JSON. No markdown, no explanations, no formatting.

Analyze these {len(images)} soccer frames (0 to {len(images)-1}).

Detect events: Goals, Shots, Fouls, Cards, Corners, Penalties.

CRITICAL: Your ENTIRE response must be ONLY this JSON structure with NO additional text:

{{"chunk": 1, "start_frame": 0, "end_frame": {len(images)-1}, "events": [{{"event_type": "shot", "frame_number": 3, "timestamp": "00:03", "team": "home", "player": "#10", "confidence": 0.9, "description": "shot on goal"}}, {{"event_type": "corner", "frame_number": 5, "timestamp": "00:05", "team": "away", "player": "#7", "confidence": 0.8, "description": "corner kick"}}]}}

Do NOT use markdown. Do NOT add explanations. Output ONLY the JSON object."""

# Llama format: images before text
messages = [
    {
        "role": "user",
        "content": [{"type": "image"}] * len(images) +
                  [{"type": "text", "text": prompt}]
    }
]

input_text = processor.apply_chat_template(messages, add_generation_prompt=True)

inputs = processor(
    text=input_text,
    images=images,
    return_tensors="pt"
).to(model.device)

print("⏳ Generating response...")
start_time = datetime.now()

with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=4096)  # Increased from 2048

elapsed = (datetime.now() - start_time).total_seconds()
output_text = processor.decode(output[0], skip_special_tokens=True)

# CRITICAL FIX: Llama includes the full prompt in output, extract only assistant response
if "assistant" in output_text.lower():
    # Split on common markers
    parts = output_text.split("assistant")
    if len(parts) > 1:
        output_text = parts[-1].strip()

# Also try splitting on the user prompt if still present
if "Analyze these" in output_text:
    # Find where the assistant response starts (after the prompt)
    json_start = output_text.find("{")
    if json_start != -1:
        output_text = output_text[json_start:]

print(f"✅ Generation complete ({elapsed:.1f}s)")

# Parse result
print("\n" + "="*80)
print("📊 RESULTS")
print("="*80)

print(f"\n📄 Raw output (first 1000 chars):")
print(output_text[:1000])
print("\n" + "-"*80)

try:
    # First try to find JSON object directly
    json_start = output_text.find('{"chunk"')

    if json_start != -1:
        # Extract from start of JSON to end
        json_text = output_text[json_start:]

        # Check if JSON is complete (ends with })
        if not json_text.rstrip().endswith('}'):
            print("\n⚠️  JSON appears truncated, attempting repair...")
            # Try to close the JSON properly
            if '"events":' in json_text:
                # Find the last complete event
                last_brace = json_text.rfind('}')
                if last_brace != -1:
                    json_text = json_text[:last_brace+1] + ']}'  # Close events array and main object

        result = json.loads(json_text)

        print(f"\n✅ JSON parsed successfully!")
        print(f"   Events detected: {len(result.get('events', []))}")

        if result.get('events'):
            print("\n📋 Events found:")
            for event in result['events']:
                print(f"   • {event.get('event_type', 'unknown')} at frame {event.get('frame_number', 'N/A')}")

        print("\n📄 Full JSON:")
        print(json.dumps(result, indent=2))
    else:
        raise json.JSONDecodeError("No JSON object found", output_text, 0)

except (json.JSONDecodeError, ValueError) as e:
    print(f"\n⚠️  Direct JSON extraction failed: {e}")
    print("\n🔄 Attempting to parse numbered list format...")

    # Llama returns numbered list, extract JSON objects
    import re
    events = []

    # Find all JSON objects in the text
    json_objects = re.findall(r'\{[^}]+\}', output_text)

    for obj_str in json_objects:
        try:
            event = json.loads(obj_str)
            events.append(event)
        except:
            continue

    if events:
        result = {
            "chunk": 1,
            "start_frame": 0,
            "end_frame": len(images) - 1,
            "events": events
        }

        print(f"✅ Parsed {len(events)} events from numbered list")
        print("\n📋 Sample events:")
        for event in events[:3]:
            print(f"   • {event.get('event_type', 'unknown')} at frame {event.get('frame_number', 'N/A')}")

        print(f"\n📄 Reconstructed JSON:")
        print(json.dumps(result, indent=2))
    else:
        print(f"❌ Could not extract any valid JSON objects")
        print(f"❌ Llama is not following JSON format instructions")
        result = {"error": "No valid JSON found", "events": []}

# Cleanup
print("\n🧹 Cleaning up...")
shutil.rmtree(Path(frame_paths[0]).parent, ignore_errors=True)
gc.collect()
torch.cuda.empty_cache()

print("\n" + "="*80)
print("✅ LLAMA TEST COMPLETE")
print("="*80)
print(f"⏱️  Time for {len(images)} frames: {elapsed:.1f}s")
print(f"💾 VRAM used: {torch.cuda.memory_allocated()/1024**3:.1f} GB")
