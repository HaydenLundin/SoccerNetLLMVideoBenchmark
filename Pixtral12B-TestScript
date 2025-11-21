# ============================================================================
# PIXTRAL-12B TEST SCRIPT
# ============================================================================
"""
Tests Pixtral-12B on ONE video to verify it works.
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
# LOAD PIXTRAL MODEL
# ============================================================================

print("\n" + "="*80)
print("🤖 LOADING PIXTRAL-12B")
print("="*80)

from transformers import LlavaForConditionalGeneration, AutoProcessor

model = LlavaForConditionalGeneration.from_pretrained(
    "mistral-community/pixtral-12b",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("mistral-community/pixtral-12b")

print(f"✅ Pixtral loaded (VRAM: {torch.cuda.memory_allocated()/1024**3:.1f} GB)")

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

# Take only first chunk for testing
chunks = chunk_frames(frame_paths, FRAMES_PER_CHUNK)
print(f"📦 Total chunks: {len(chunks)}")

if len(chunks) == 0:
    print("❌ No chunks created - no frames extracted")
    exit()

print(f"🧪 Testing with FIRST chunk only ({len(chunks[0])} frames)")

test_chunk = chunks[0]
images = load_images(test_chunk)

print(f"\n🤖 Running Pixtral inference on {len(images)} images...")

# Build prompt
prompt = f"""Analyze these {len(images)} consecutive soccer game frames (chunk 1/{len(chunks)}).
Frame numbers: 0 to {len(images)-1}

Detect ALL events in this time segment:
Goals, Shots, Fouls, Yellow/Red cards, Substitutions, Corners, Throw-ins, Penalties, Offsides, Clearances, Key passes, Free-kicks, Kick-offs

For each event:
{{"event_type": "goal", "frame_number": 0, "timestamp": "MM:SS", "team": "home/away", "player": "#10", "confidence": 0.9, "description": "brief"}}

Return ONLY JSON:
{{"chunk": 1, "start_frame": 0, "end_frame": {len(images)-1}, "events": [...]}}"""

# Pixtral format
chat = [
    {
        "role": "user",
        "content": [{"type": "text", "content": prompt}] +
                  [{"type": "image"} for _ in images]
    }
]

text_prompt = processor.apply_chat_template(chat, add_generation_prompt=True)

inputs = processor(
    text=text_prompt,
    images=images,
    return_tensors="pt"
).to(model.device)

# Convert to bfloat16 to match model dtype
if hasattr(inputs, 'pixel_values') and inputs.pixel_values is not None:
    inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)
elif 'pixel_values' in inputs:
    inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)

print("⏳ Generating response...")
start_time = datetime.now()

with torch.no_grad():
    generate_ids = model.generate(**inputs, max_new_tokens=2048)

elapsed = (datetime.now() - start_time).total_seconds()
output_text = processor.batch_decode(
    generate_ids,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False
)[0]

print(f"✅ Generation complete ({elapsed:.1f}s)")

# Parse result
print("\n" + "="*80)
print("📊 RESULTS")
print("="*80)

try:
    if '```json' in output_text:
        json_text = output_text.split('```json')[1].split('```')[0]
    elif '```' in output_text:
        json_text = output_text.split('```')[1].split('```')[0]
    else:
        json_text = output_text

    result = json.loads(json_text.strip())

    print(f"\n✅ JSON parsed successfully")
    print(f"   Events detected: {len(result.get('events', []))}")

    if result.get('events'):
        print("\n📋 Sample events:")
        for event in result['events'][:3]:
            print(f"   • {event.get('event_type', 'unknown')} at frame {event.get('frame_number', 'N/A')}")

    print("\n📄 Full JSON:")
    print(json.dumps(result, indent=2))

except json.JSONDecodeError as e:
    print(f"\n❌ JSON parse error: {e}")
    print(f"\n📄 Raw output (first 500 chars):")
    print(output_text[:500])

# Cleanup
print("\n🧹 Cleaning up...")
shutil.rmtree(Path(frame_paths[0]).parent, ignore_errors=True)
gc.collect()
torch.cuda.empty_cache()

print("\n" + "="*80)
print("✅ PIXTRAL TEST COMPLETE")
print("="*80)
print(f"⏱️  Time for {len(images)} frames: {elapsed:.1f}s")
print(f"💾 VRAM used: {torch.cuda.memory_allocated()/1024**3:.1f} GB")
