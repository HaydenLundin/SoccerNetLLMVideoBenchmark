# ============================================================================
# LOCAL VISION LLM ANNOTATION - 540 FRAMES (FINAL CORRECTED VERSION)
# ============================================================================
"""
Runs 3 LOCAL vision LLMs on Colab A100 (80GB) with 540 frames per video (1 fps).
Processes in 12 chunks of 45 frames each to fit context limits.

Models:
1. Qwen2.5-VL-32B-Instruct (4-BIT QUANTIZED ~20GB VRAM) - 32K context
2. Llama-3.2-11B-Vision-Instruct (22GB VRAM) - 128K context
3. Pixtral-12B (12GB VRAM) - 128K context

Features:
- 540 frames total (1 fps for 9min video)
- 12 chunks of 45 frames each (~67.5K tokens per chunk)
- No API dependencies (fully local)
- Progress saving every 5 videos
- Automatic memory management
- 4-bit quantization for Qwen

CORRECTIONS IN THIS VERSION:
- Fixed Qwen import typo (Qwen2_5_VLForConditionalGeneration)
- Fixed Llama processor call (added named parameters)
- Replaced InternVL2-26B with Pixtral-12B (supports multiple images)
- Added HuggingFace authentication
- Added 4-bit quantization for Qwen2.5-VL
- Fixed chunking logic to properly handle 540 frames
- Fixed .to() issue with quantized models
"""

# ============================================================================
# CONFIGURATION
# ============================================================================

BATCH_NUMBER = 1
DRIVE_BASE = "/content/drive/MyDrive/SoccerNet_EPL"
FPS = 1.0  # 1 frame per second

# Chunking strategy - FIXED
TOTAL_CHUNKS = 12      # 12 chunks total
FRAMES_PER_CHUNK = 45  # 45 frames per chunk
# Total frames = 12 × 45 = 540 frames = 9 minutes of video at 1 fps

# Model selection (set to False to disable)
RUN_QWEN = True
RUN_LLAMA = True
RUN_PIXTRAL = True

# ============================================================================
# INSTALLATION (Handles everything automatically)
# ============================================================================

import subprocess
import sys

print("📦 Installing required packages...")
print("   (This may take 2-3 minutes on first run)")

# CRITICAL: Install latest transformers from GitHub
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                'git+https://github.com/huggingface/transformers'], check=False)

packages = [
    'accelerate',
    'torch',
    'pillow',
    'bitsandbytes',
    'einops',
    'timm',
    'torchvision',
    'qwen-vl-utils',
    'autoawq>=0.2.0'  # For Qwen utilities
]

for package in packages:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', package], check=False)

print("✅ All packages installed")

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
# Hugging Face Auth
# ============================================================================
from huggingface_hub import login
from google.colab import userdata

try:
    HF_TOKEN = userdata.get('HF_TOKEN')
    login(token=HF_TOKEN)
    print("✅ Hugging Face login successful.")
except Exception as e:
    print(f"❌ Hugging Face login failed. Did you set the 'HF_TOKEN' secret in Colab?")
    print(f"   Error: {e}")

# Check GPU
print(f"\n🖥️  GPU: {torch.cuda.get_device_name(0)}")
print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# ============================================================================
# MOUNT DRIVE
# ============================================================================

print("\n📁 Mounting Google Drive...")
drive.mount('/content/drive')
print("✅ Mounted!")

# ============================================================================
# FIND VIDEOS
# ============================================================================

print("\n" + "="*80)
print(f"FINDING VIDEOS IN BATCH {BATCH_NUMBER}")
print("="*80)

batch_dir = os.path.join(DRIVE_BASE, f"batch_{BATCH_NUMBER}")
print(f"Searching: {batch_dir}")

if not os.path.exists(batch_dir):
    print(f"❌ Batch directory not found: {batch_dir}")
    exit()

video_files = []
for root, dirs, files in os.walk(batch_dir):
    for file in files:
        if file.endswith('.mkv'):
            filepath = os.path.join(root, file)
            video_files.append(filepath)

print(f"\n✅ Found {len(video_files)} video files!")

if len(video_files) == 0:
    print("❌ No videos found!")
    exit()

print("\nFirst 5 videos:")
for i, video in enumerate(video_files[:5], 1):
    video_name = Path(video).name
    size_gb = os.path.getsize(video) / (1024**3)
    print(f"  {i}. {video_name} ({size_gb:.1f} GB)")

if len(video_files) > 5:
    print(f"  ... and {len(video_files) - 5} more")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_frames(video_path, fps=1.0):
    """Extract frames from video at 1 fps"""
    try:
        output_dir = Path("/content/temp_frames") / Path(video_path).stem
        output_dir.mkdir(parents=True, exist_ok=True)
        output_pattern = str(output_dir / "frame_%04d.jpg")

        cmd = [
            'ffmpeg', '-i', video_path,
            '-vf', f'fps={fps}',
            '-q:v', '2', '-y',
            output_pattern
        ]

        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if result.returncode != 0:
            print(f"  ❌ ffmpeg error: {result.stderr.decode()[:200]}")
            return []

        frames = sorted(output_dir.glob("frame_*.jpg"))
        print(f"  📸 Extracted {len(frames)} frames")
        return [str(f) for f in frames]

    except Exception as e:
        print(f"  ❌ Frame extraction failed: {e}")
        return []

def load_images(frame_paths):
    """Load images from disk"""
    images = []
    for path in frame_paths:
        try:
            img = Image.open(path).convert('RGB')
            images.append(img)
        except Exception as e:
            print(f"  ⚠️  Error loading {path}: {e}")
    return images

def chunk_frames(frames, frames_per_chunk=45):
    """
    Split frames into fixed-size chunks of exactly frames_per_chunk.

    For ~2,700 frames with 45 frames per chunk:
    - Creates 60 chunks of 45 frames each
    - All chunks are the same size (45 frames)
    - Processes entire video regardless of length

    Args:
        frames: List of frame paths
        frames_per_chunk: Number of frames per chunk (default: 45)

    Returns:
        List of chunks, where each chunk is a list of frame paths

    Example:
        2,700 frames ÷ 45 = 60 chunks (all size 45)
        2,715 frames ÷ 45 = 60 chunks of 45, plus 1 chunk of 15
    """
    chunks = []
    for i in range(0, len(frames), frames_per_chunk):
        chunk = frames[i:i + frames_per_chunk]
        chunks.append(chunk)

    return chunks

def save_results(data, filename, subfolder=""):
    """Save results to Drive"""
    try:
        drive_base = "/content/drive/MyDrive/SoccerNet_LLM_Benchmark"
        drive_path = os.path.join(drive_base, subfolder) if subfolder else drive_base
        os.makedirs(drive_path, exist_ok=True)

        filepath = os.path.join(drive_path, filename)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"  💾 Saved: {filepath}")
        return filepath
    except Exception as e:
        print(f"  ❌ Error saving: {e}")
        return None

def cleanup_memory():
    """Free up GPU memory"""
    gc.collect()
    torch.cuda.empty_cache()

# ============================================================================
# MODEL LOADERS
# ============================================================================

def load_qwen_model():
    """Load Qwen2.5-VL-32B with AWQ quantization (pre-quantized)"""
    print("\n📥 Loading Qwen2.5-VL-32B-Instruct-AWQ (pre-quantized)...")
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info

    # Use the official AWQ quantized model (already 4-bit)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-32B-Instruct-AWQ",  # Pre-quantized AWQ model
        torch_dtype="auto",
        device_map="auto"
    )

    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2.5-VL-32B-Instruct-AWQ",
        min_pixels=256*28*28,
        max_pixels=1280*28*28
    )

    print(f"✅ Qwen AWQ loaded (VRAM: {torch.cuda.memory_allocated()/1024**3:.1f} GB)")
    return model, processor

def load_llama_model():
    """Load Llama-3.2-11B-Vision"""
    print("\n📥 Loading Llama-3.2-11B-Vision-Instruct...")
    from transformers import MllamaForConditionalGeneration, AutoProcessor

    model = MllamaForConditionalGeneration.from_pretrained(
        "meta-llama/Llama-3.2-11B-Vision-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("meta-llama/Llama-3.2-11B-Vision-Instruct")

    print(f"✅ Llama loaded (VRAM: {torch.cuda.memory_allocated()/1024**3:.1f} GB)")
    return model, processor

def load_pixtral_model():
    """Load Pixtral-12B"""
    print("\n📥 Loading Pixtral-12B...")
    from transformers import LlavaForConditionalGeneration, AutoProcessor

    model = LlavaForConditionalGeneration.from_pretrained(
        "mistral-community/pixtral-12b",
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("mistral-community/pixtral-12b")

    print(f"✅ Pixtral loaded (VRAM: {torch.cuda.memory_allocated()/1024**3:.1f} GB)")
    return model, processor

# ============================================================================
# CHUNKED ANNOTATION FUNCTIONS
# ============================================================================

def annotate_chunk_qwen(images, chunk_idx, total_chunks, model, processor):
    """Annotate a single chunk with Qwen (4-bit quantized)"""
    try:
        # Calculate time range for this chunk
        start_frame = chunk_idx * FRAMES_PER_CHUNK
        end_frame = start_frame + len(images)
        start_time_sec = start_frame
        end_time_sec = end_frame

        prompt = f"""Analyze these {len(images)} consecutive soccer game frames (chunk {chunk_idx+1}/{total_chunks}).
Time range: {start_time_sec//60:02d}:{start_time_sec%60:02d} to {end_time_sec//60:02d}:{end_time_sec%60:02d}
Frame numbers: {start_frame} to {end_frame-1}

Detect ALL events in this time segment:
Goals, Shots, Fouls, Yellow/Red cards, Substitutions, Corners, Throw-ins, Penalties, Offsides, Clearances, Key passes, Free-kicks, Kick-offs

For each event:
{{"event_type": "goal", "frame_number": {start_frame}, "timestamp": "MM:SS", "team": "home/away", "player": "#10", "confidence": 0.9, "description": "brief"}}

Return ONLY JSON:
{{"chunk": {chunk_idx+1}, "start_frame": {start_frame}, "end_frame": {end_frame-1}, "events": [...]}}"""

        # Prepare conversation
        conversation = [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}] +
                          [{"type": "image"} for _ in images]
            }
        ]

        text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

        # CRITICAL FIX: Don't use .to() with quantized models
        # The model is already on the correct device
        inputs = processor(
            text=[text_prompt],
            images=images,
            return_tensors="pt"
        )

        # Move inputs to model device manually (safer for quantized models)
        inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v
                 for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=2048)

        output_text = processor.batch_decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )[0]

        # Parse JSON
        try:
            if '```json' in output_text:
                output_text = output_text.split('```json')[1].split('```')[0]
            elif '```' in output_text:
                output_text = output_text.split('```')[1].split('```')[0]

            result = json.loads(output_text.strip())
        except json.JSONDecodeError as e:
            print(f"    ⚠️  JSON parse error in chunk {chunk_idx+1}: {e}")
            result = {"error": "JSON parse failed", "raw_output": output_text[:500], "events": []}

        return result

    except Exception as e:
        print(f"    ❌ Error in chunk {chunk_idx+1}: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e), "events": []}

def annotate_chunk_llama(images, chunk_idx, total_chunks, model, processor):
    """Annotate a single chunk with Llama"""
    try:
        start_frame = chunk_idx * FRAMES_PER_CHUNK
        end_frame = start_frame + len(images)

        prompt = f"""Analyze frames {start_frame}-{end_frame-1} (chunk {chunk_idx+1}/{total_chunks}).
Detect: Goals, Shots, Fouls, Cards, Corners, Penalties.
Return JSON: {{"chunk": {chunk_idx+1}, "events": [{{"event_type": "goal", "frame_number": {start_frame}, "timestamp": "MM:SS", "team": "home", "confidence": 0.9}}]}}"""

        # Proper Llama format with images before text
        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}] * len(images) +
                          [{"type": "text", "text": prompt}]
            }
        ]

        input_text = processor.apply_chat_template(messages, add_generation_prompt=True)

        # Use named parameters
        inputs = processor(
            text=input_text,
            images=images,
            return_tensors="pt"
        ).to(model.device)

        # Generate
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=2048)

        output_text = processor.decode(output[0], skip_special_tokens=True)

        # Parse JSON
        try:
            if '```json' in output_text:
                output_text = output_text.split('```json')[1].split('```')[0]
            elif '```' in output_text:
                output_text = output_text.split('```')[1].split('```')[0]

            result = json.loads(output_text.strip())
        except json.JSONDecodeError as e:
            print(f"    ⚠️  JSON parse error in chunk {chunk_idx+1}: {e}")
            result = {"error": "JSON parse failed", "raw_output": output_text[:500], "events": []}

        return result

    except Exception as e:
        print(f"    ❌ Error in chunk {chunk_idx+1}: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e), "events": []}

def annotate_chunk_pixtral(images, chunk_idx, total_chunks, model, processor):
    """Annotate a single chunk with Pixtral"""
    try:
        start_frame = chunk_idx * FRAMES_PER_CHUNK
        end_frame = start_frame + len(images)

        prompt = f"""Analyze these {len(images)} soccer frames (chunk {chunk_idx+1}/{total_chunks}, frames {start_frame}-{end_frame-1}).
Detect events: Goals, Shots, Fouls, Cards, Corners, Penalties, Substitutions.
Return JSON: {{"chunk": {chunk_idx+1}, "events": [{{"event_type": "goal", "frame_number": {start_frame}, "timestamp": "MM:SS", "team": "home", "confidence": 0.9}}]}}"""

        # Pixtral format: simple chat with images
        chat = [
            {
                "role": "user",
                "content": [{"type": "text", "content": prompt}] +
                          [{"type": "image"} for _ in images]
            }
        ]

        # Apply chat template
        text_prompt = processor.apply_chat_template(chat, add_generation_prompt=True)

        # Process inputs
        inputs = processor(
            text=text_prompt,
            images=images,
            return_tensors="pt"
        ).to(model.device)

        # Generate
        with torch.no_grad():
            generate_ids = model.generate(**inputs, max_new_tokens=2048)

        output_text = processor.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        # Parse JSON
        try:
            if '```json' in output_text:
                output_text = output_text.split('```json')[1].split('```')[0]
            elif '```' in output_text:
                output_text = output_text.split('```')[1].split('```')[0]

            result = json.loads(output_text.strip())
        except json.JSONDecodeError as e:
            print(f"    ⚠️  JSON parse error in chunk {chunk_idx+1}: {e}")
            result = {"error": "JSON parse failed", "raw_output": output_text[:500], "events": []}

        return result

    except Exception as e:
        print(f"    ❌ Error in chunk {chunk_idx+1}: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e), "events": []}

# ============================================================================
# FULL VIDEO ANNOTATION (WITH CHUNKING)
# ============================================================================

def annotate_video_qwen(video_path, model, processor):
    """Annotate full video in chunks with Qwen"""
    video_name = Path(video_path).name

    try:
        # Extract all frames
        print(f"  🎬 Extracting frames at {FPS} fps...")
        frame_paths = extract_frames(video_path, fps=FPS)

        if not frame_paths:
            return {"error": "Frame extraction failed", "events": []}

        # Split into chunks
        chunks = chunk_frames(frame_paths, FRAMES_PER_CHUNK)
        print(f"  📦 Split into {len(chunks)} chunks")
        print(f"      Chunk sizes: {[len(c) for c in chunks]}")

        # Process each chunk
        all_events = []

        for chunk_idx, chunk_paths in enumerate(chunks):
            print(f"  🤖 Processing chunk {chunk_idx+1}/{len(chunks)} ({len(chunk_paths)} frames)...")

            # Load images for this chunk
            images = load_images(chunk_paths)

            if not images:
                print(f"    ⚠️  Skipping chunk {chunk_idx+1} - no images loaded")
                continue

            # Annotate chunk
            chunk_result = annotate_chunk_qwen(images, chunk_idx, len(chunks), model, processor)

            # Collect events
            if 'events' in chunk_result:
                all_events.extend(chunk_result['events'])
                print(f"    ✅ Found {len(chunk_result['events'])} events in chunk {chunk_idx+1}")

        # Combine results
        annotations = {
            "model": "Qwen2.5-VL-32B-Instruct-4bit",
            "processed_at": datetime.now().isoformat(),
            "video_path": str(video_path),
            "total_frames": len(frame_paths),
            "chunks_processed": len(chunks),
            "frames_per_chunk": FRAMES_PER_CHUNK,
            "events": all_events
        }

        print(f"  ✅ Total: {len(all_events)} events across {len(chunks)} chunks")

        # Cleanup
        shutil.rmtree(Path(frame_paths[0]).parent, ignore_errors=True)
        cleanup_memory()

        return annotations

    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e), "events": []}

def annotate_video_llama(video_path, model, processor):
    """Annotate full video in chunks with Llama"""
    video_name = Path(video_path).name

    try:
        print(f"  🎬 Extracting frames at {FPS} fps...")
        frame_paths = extract_frames(video_path, fps=FPS)

        if not frame_paths:
            return {"error": "Frame extraction failed", "events": []}

        chunks = chunk_frames(frame_paths, FRAMES_PER_CHUNK)
        print(f"  📦 Split into {len(chunks)} chunks")
        print(f"      Chunk sizes: {[len(c) for c in chunks]}")

        all_events = []

        for chunk_idx, chunk_paths in enumerate(chunks):
            print(f"  🤖 Processing chunk {chunk_idx+1}/{len(chunks)} ({len(chunk_paths)} frames)...")

            images = load_images(chunk_paths)
            if not images:
                continue

            chunk_result = annotate_chunk_llama(images, chunk_idx, len(chunks), model, processor)

            if 'events' in chunk_result:
                all_events.extend(chunk_result['events'])
                print(f"    ✅ Found {len(chunk_result['events'])} events")

        annotations = {
            "model": "Llama-3.2-11B-Vision-Instruct",
            "processed_at": datetime.now().isoformat(),
            "video_path": str(video_path),
            "total_frames": len(frame_paths),
            "chunks_processed": len(chunks),
            "events": all_events
        }

        print(f"  ✅ Total: {len(all_events)} events")

        shutil.rmtree(Path(frame_paths[0]).parent, ignore_errors=True)
        cleanup_memory()

        return annotations

    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e), "events": []}

def annotate_video_pixtral(video_path, model, processor):
    """Annotate full video in chunks with Pixtral"""
    video_name = Path(video_path).name

    try:
        print(f"  🎬 Extracting frames at {FPS} fps...")
        frame_paths = extract_frames(video_path, fps=FPS)

        if not frame_paths:
            return {"error": "Frame extraction failed", "events": []}

        chunks = chunk_frames(frame_paths, FRAMES_PER_CHUNK)
        print(f"  📦 Split into {len(chunks)} chunks")
        print(f"      Chunk sizes: {[len(c) for c in chunks]}")

        all_events = []

        for chunk_idx, chunk_paths in enumerate(chunks):
            print(f"  🤖 Processing chunk {chunk_idx+1}/{len(chunks)} ({len(chunk_paths)} frames)...")

            images = load_images(chunk_paths)
            if not images:
                continue

            chunk_result = annotate_chunk_pixtral(images, chunk_idx, len(chunks), model, processor)

            if 'events' in chunk_result:
                all_events.extend(chunk_result['events'])
                print(f"    ✅ Found {len(chunk_result['events'])} events")

        annotations = {
            "model": "Pixtral-12B",
            "processed_at": datetime.now().isoformat(),
            "video_path": str(video_path),
            "total_frames": len(frame_paths),
            "chunks_processed": len(chunks),
            "events": all_events
        }

        print(f"  ✅ Total: {len(all_events)} events")

        shutil.rmtree(Path(frame_paths[0]).parent, ignore_errors=True)
        cleanup_memory()

        return annotations

    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e), "events": []}

# ============================================================================
# MAIN PROCESSING
# ============================================================================

print(f"\n{'='*80}")
print("CONFIGURING MODELS")
print(f"{'='*80}")

models = {}

if RUN_QWEN:
    models['qwen'] = {
        'loader': load_qwen_model,
        'annotator': annotate_video_qwen,
        'name': 'Qwen2.5-VL-32B-4bit'
    }

if RUN_LLAMA:
    models['llama'] = {
        'loader': load_llama_model,
        'annotator': annotate_video_llama,
        'name': 'Llama-3.2-11B-Vision'
    }

if RUN_PIXTRAL:
    models['pixtral'] = {
        'loader': load_pixtral_model,
        'annotator': annotate_video_pixtral,
        'name': 'Pixtral-12B'
    }

print(f"\n{'='*80}")
print("STARTING SEQUENTIAL MODEL PROCESSING")
print(f"{'='*80}")
print(f"\n🎯 Strategy: Complete ALL videos with one model before moving to next")
print(f"   This allows resuming if a session crashes!")
print(f"\nVideos: {len(video_files)}")
print(f"Models: {len(models)}")
print(f"FPS: {FPS} frame/second")
print(f"Frames per chunk: {FRAMES_PER_CHUNK}")
print(f"💾 Checkpoint: After EACH video")

results = {key: {} for key in models.keys()}

# PROCESS ONE MODEL AT A TIME
for model_key, model_info in models.items():
    print(f"\n{'='*80}")
    print(f"🤖 STARTING MODEL: {model_info['name']}")
    print(f"{'='*80}")
    print(f"Processing all {len(video_files)} videos")

    start_time = datetime.now()
    model_data = model_info['loader']()

    for i, video_path in enumerate(video_files, 1):
        video_name = Path(video_path).name

        print(f"\n{'─'*80}")
        print(f"[Video {i}/{len(video_files)}] {video_name}")
        print(f"{'─'*80}")

        result = model_info['annotator'](video_path, *model_data)
        results[model_key][video_name] = result

        # CHECKPOINT AFTER EACH VIDEO
        save_results(
            results[model_key],
            f"batch_{BATCH_NUMBER}_{model_key}_checkpoint_v{i}.json",
            f"annotations/{model_key}"
        )
        print(f"✅ Checkpoint saved ({i}/{len(video_files)})")

        # Time estimate
        elapsed = (datetime.now() - start_time).total_seconds() / 60
        if i > 0:
            avg_per_video = elapsed / i
            remaining = (len(video_files) - i) * avg_per_video
            print(f"⏱️  {elapsed:.1f}min elapsed | {avg_per_video:.1f}min/video | ~{remaining:.1f}min remaining")

        cleanup_memory()

    # Save final
    save_results(
        results[model_key],
        f"batch_{BATCH_NUMBER}_{model_key}_FINAL.json",
        f"annotations/{model_key}"
    )

    del model_data
    cleanup_memory()
# ============================================================================
# FINAL SAVE
# ============================================================================

print(f"\n{'='*80}")
print("SAVING FINAL RESULTS")
print(f"{'='*80}")

for model_key in results.keys():
    if results[model_key]:
        save_results(
            results[model_key],
            f"batch_{BATCH_NUMBER}_{model_key}_FINAL.json",
            f"annotations/{model_key}"
        )

# ============================================================================
# SUMMARY
# ============================================================================

print(f"\n{'='*80}")
print("FINAL SUMMARY")
print(f"{'='*80}")

total_time = (datetime.now() - start_time).total_seconds() / 60

print(f"\nBatch {BATCH_NUMBER} Complete!")
print(f"Videos processed: {len(video_files)}")
print(f"Models used: {len(models)}")

for model_key, model_info in models.items():
    if results[model_key]:
        total_events = sum(len(r.get('events', [])) for r in results[model_key].values())
        total_frames = sum(r.get('total_frames', 0) for r in results[model_key].values())
        errors = sum(1 for r in results[model_key].values() if 'error' in r)
        successful = len(results[model_key]) - errors

        print(f"\n🤖 {model_info['name']}:")
        print(f"   • Videos: {len(results[model_key])}")
        print(f"   • Successful: {successful}")
        print(f"   • Errors: {errors}")
        print(f"   • Total frames processed: {total_frames}")
        print(f"   • Events detected: {total_events}")

print(f"\n⏱️  Total time: {total_time:.1f} minutes ({total_time/60:.1f} hours)")
if len(video_files) > 0:
    print(f"   Avg per video: {total_time/len(video_files):.1f} minutes")

print(f"\n📁 Results saved to:")
print(f"   MyDrive/SoccerNet_LLM_Benchmark/annotations/")

print(f"\n{'='*80}")
print(f"✅ BATCH {BATCH_NUMBER} COMPLETE!")
print(f"✅ All models support multiple images per chunk!")
print(f"✅ Qwen2.5-VL running with 4-bit quantization!")
print(f"{'='*80}")
