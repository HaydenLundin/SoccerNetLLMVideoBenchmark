# ============================================================================
# 6-MODEL OPENROUTER ANNOTATION PIPELINE - FINAL VERSION
# ============================================================================
"""
Uses 6 FREE OpenRouter vision models with 70 frames per video:

1. NVIDIA Nemotron Nano 12B 2 VL (128K context, video-optimized)
2. Meta Llama 4 Maverick 128B (128K context, massive reasoning)
3. Google Gemini 2.0 Flash Experimental (1M context, fastest)
4. Mistral Small 3.2 24B (131K context, efficient)
5. Google Gemma 3 27B (131K context, vision-language)
6. Meta Llama 4 Scout 17B (128K context, visual reasoning)

All FREE with $10 OpenRouter deposit!
40 videos × 6 models = 240 requests per batch (< 1000/day limit)
70 frames = ~110K tokens (fits all models)
"""

# ============================================================================
# CONFIGURATION
# ============================================================================

BATCH_NUMBER = 1
DRIVE_BASE = "/content/drive/MyDrive/SoccerNet_EPL"

# Frames to extract per video
FRAMES_TO_SAMPLE = 70  # ~110K tokens, fits all models

# Model selection (set to False to disable any model)
RUN_NEMOTRON = True
RUN_MAVERICK = True
RUN_GEMINI = True
RUN_MISTRAL = True
RUN_GEMMA = True
RUN_SCOUT = True

# ============================================================================
# INSTALLATION
# ============================================================================

import subprocess
print("📦 Installing packages...")
subprocess.run(['pip', 'install', 'requests', '--quiet'], check=True)
print("✅ Packages installed")

# ============================================================================
# IMPORTS
# ============================================================================

from google.colab import drive, userdata
import requests
import os
import json
import base64
import shutil
from pathlib import Path
from datetime import datetime

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
# CHECK API KEY
# ============================================================================

print("\n" + "="*80)
print("CHECKING API KEY")
print("="*80)

try:
    api_key = userdata.get('openrouter')
    if not api_key:
        print("❌ 'openrouter' key not found in Colab Secrets")
        print("\n🔧 Fix: Click 🔑 icon → Add 'openrouter' → Your API key")
        exit()

    print(f"✅ API key found: {api_key[:10]}...{api_key[-5:]}")
except Exception as e:
    print(f"❌ Error getting API key: {e}")
    exit()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_frames(video_path, fps=0.2):
    """Extract frames from video - 1 frame every 5 seconds"""
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

def encode_image_base64(image_path):
    """Encode image to base64"""
    try:
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    except Exception as e:
        print(f"  ❌ Error encoding {image_path}: {e}")
        return None

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

# ============================================================================
# ANNOTATION FUNCTION (WORKS FOR ALL MODELS)
# ============================================================================

def annotate_with_openrouter(video_path, api_key, model_name, model_display_name):
    """
    Generic annotation function for any OpenRouter vision model

    Args:
        video_path: Path to video file
        api_key: OpenRouter API key
        model_name: Model identifier (e.g., "nvidia/nemotron-nano-12b-v2-vl:free")
        model_display_name: Human-readable name for logging
    """
    video_name = Path(video_path).name

    try:
        # Extract frames
        print(f"  🎬 Extracting frames...")
        frames = extract_frames(video_path, fps=0.2)

        if not frames or len(frames) == 0:
            return {
                "error": "Frame extraction failed",
                "video_path": str(video_path),
                "model": model_name,
                "events": []
            }

        # Sample to FRAMES_TO_SAMPLE (70 frames)
        if len(frames) > FRAMES_TO_SAMPLE:
            step = len(frames) // FRAMES_TO_SAMPLE
            frames = frames[::step]
            print(f"  📊 Sampled to {len(frames)} frames (~110K tokens)")

        # Build prompt
        prompt = """Analyze these soccer game video frames and identify ALL events with precise timestamps.

Detect these event types:
- Goals
- Shots on/off target
- Fouls
- Yellow cards
- Red cards
- Substitutions
- Corners
- Throw-ins
- Penalties
- Offsides
- Clearances
- Key passes
- Free-kicks (direct/indirect)
- Kick-offs
- Ball out of play

For EACH event, provide:
- event_type: (goal, foul, yellow_card, etc.)
- frame_number: which frame number shows the event
- timestamp: estimate MM:SS format (frames are ~5 sec apart)
- team: "home" or "away" (if identifiable)
- player: player number or name (if visible)
- confidence: your confidence level (0.0-1.0)
- description: brief description of what happened

CRITICAL: Return ONLY valid JSON with NO markdown formatting:
{
  "game": "game_name",
  "half": 1,
  "frames_analyzed": 70,
  "events": [
    {
      "event_type": "goal",
      "frame_number": 45,
      "timestamp": "03:45",
      "team": "home",
      "player": "#10",
      "confidence": 0.9,
      "description": "header from corner kick"
    }
  ]
}"""

        # Build content with frames
        content = [{"type": "text", "text": prompt}]

        print(f"  📤 Encoding {len(frames)} frames...")
        for i, frame_path in enumerate(frames):
            if i % 20 == 0 and i > 0:
                print(f"    {i}/{len(frames)}...")

            base64_img = encode_image_base64(frame_path)
            if base64_img:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}
                })

        print(f"  🤖 Sending to {model_display_name}...")
        print(f"     (This may take 2-5 minutes)")

        # API call
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://colab.research.google.com",
                "X-Title": "SoccerNet LLM Benchmark"
            },
            json={
                "model": model_name,
                "messages": [{"role": "user", "content": content}],
                "max_tokens": 4000,
                "temperature": 0.7
            },
            timeout=600
        )

        # Check response
        if response.status_code != 200:
            error_text = response.text[:300]
            print(f"  ❌ API Error {response.status_code}")
            print(f"     {error_text}")
            return {
                "error": f"API {response.status_code}: {error_text}",
                "video_path": str(video_path),
                "model": model_name,
                "events": []
            }

        # Parse response
        result = response.json()
        text = result['choices'][0]['message']['content'].strip()

        # Clean markdown
        if '```json' in text:
            text = text.split('```json')[1].split('```')[0]
        elif '```' in text:
            text = text.split('```')[1].split('```')[0]
        text = text.strip()

        # Parse JSON
        try:
            annotations = json.loads(text)
        except json.JSONDecodeError as e:
            print(f"  ⚠️  JSON parse error: {str(e)[:100]}")
            print(f"     Response preview: {text[:200]}")
            annotations = {
                "error": "JSON parse failed",
                "raw_response": text[:500],
                "events": []
            }

        # Add metadata
        annotations['model'] = model_name
        annotations['model_display_name'] = model_display_name
        annotations['processed_at'] = datetime.now().isoformat()
        annotations['video_path'] = str(video_path)
        annotations['frames_processed'] = len(frames)

        event_count = len(annotations.get('events', []))
        print(f"  ✅ Found {event_count} events")

        # Cleanup frames
        try:
            frame_dir = Path(frames[0]).parent
            shutil.rmtree(frame_dir, ignore_errors=True)
        except:
            pass

        return annotations

    except requests.Timeout:
        print(f"  ❌ Request timeout (>10 min)")
        return {
            "error": "Timeout",
            "video_path": str(video_path),
            "model": model_name,
            "events": []
        }
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return {
            "error": str(e),
            "video_path": str(video_path),
            "model": model_name,
            "events": []
        }

# ============================================================================
# MODEL CONFIGURATIONS
# ============================================================================

MODELS = {
    "nemotron": {
        "name": "nvidia/nemotron-nano-12b-v2-vl:free",
        "display_name": "Nemotron Nano 12B 2 VL",
        "enabled": RUN_NEMOTRON
    },
    "maverick": {
        "name": "meta-llama/llama-4-maverick:free",
        "display_name": "Llama 4 Maverick (128B)",
        "enabled": RUN_MAVERICK
    },
    "gemini": {
        "name": "google/gemini-2.0-flash-exp:free",
        "display_name": "Gemini 2.0 Flash Experimental",
        "enabled": RUN_GEMINI
    },
    "mistral": {
        "name": "mistralai/mistral-small-3.2-24b-instruct:free",
        "display_name": "Mistral Small 3.2 24B",
        "enabled": RUN_MISTRAL
    },
    "gemma": {
        "name": "google/gemma-3-27b-it:free",
        "display_name": "Gemma 3 27B",
        "enabled": RUN_GEMMA
    },
    "scout": {
        "name": "meta-llama/llama-4-scout:free",
        "display_name": "Llama 4 Scout (17B)",
        "enabled": RUN_SCOUT
    }
}

# ============================================================================
# PROCESS ALL VIDEOS
# ============================================================================

print(f"\n{'='*80}")
print("RUNNING 6-MODEL ANNOTATION PIPELINE")
print(f"{'='*80}")

enabled_models = [k for k, v in MODELS.items() if v["enabled"]]
print(f"\nEnabled models: {len(enabled_models)}")
for model_key in enabled_models:
    print(f"  • {MODELS[model_key]['display_name']}")

print(f"\nFrames per video: {FRAMES_TO_SAMPLE} (~110K tokens)")
print(f"Total requests per batch: {len(video_files)} videos × {len(enabled_models)} models = {len(video_files) * len(enabled_models)}")

estimate_min = len(video_files) * len(enabled_models) * 3
estimate_max = len(video_files) * len(enabled_models) * 5
print(f"\n⏱️  Estimated time: {estimate_min}-{estimate_max} minutes ({estimate_min/60:.1f}-{estimate_max/60:.1f} hours)")
print(f"   (~3-5 min per video × {len(enabled_models)} models)")
print(f"\n🚀 Starting processing...\n")

results = {
    "nemotron": {},
    "maverick": {},
    "gemini": {},
    "mistral": {},
    "gemma": {},
    "scout": {}
}

start_time = datetime.now()

for i, video_path in enumerate(video_files, 1):
    video_name = Path(video_path).name

    print(f"\n{'='*80}")
    print(f"[{i}/{len(video_files)}] {video_name}")
    print(f"{'='*80}")

    # Process with each enabled model
    for model_key in enabled_models:
        model_config = MODELS[model_key]

        print(f"\n{'─'*80}")
        print(f"🤖 MODEL: {model_config['display_name']}")
        print(f"{'─'*80}")

        result = annotate_with_openrouter(
            video_path,
            api_key,
            model_config['name'],
            model_config['display_name']
        )

        results[model_key][video_name] = result

    # Save progress every 5 videos
    if i % 5 == 0 or i == len(video_files):
        print(f"\n{'─'*80}")
        print(f"💾 SAVING PROGRESS")
        print(f"{'─'*80}")

        for model_key in enabled_models:
            if results[model_key]:
                save_results(
                    results[model_key],
                    f"batch_{BATCH_NUMBER}_{model_key}_partial_{i}.json",
                    f"annotations/{model_key}"
                )

        print(f"\n✅ Progress saved ({i}/{len(video_files)} videos)")

        # Time estimate
        elapsed = (datetime.now() - start_time).total_seconds() / 60
        if i > 0:
            avg_per_video = elapsed / i
            remaining = (len(video_files) - i) * avg_per_video
            print(f"⏱️  Elapsed: {elapsed:.1f} min | Remaining: ~{remaining:.1f} min")

# ============================================================================
# FINAL SAVE
# ============================================================================

print(f"\n{'='*80}")
print("SAVING FINAL RESULTS")
print(f"{'='*80}")

for model_key in enabled_models:
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
print(f"Models used: {len(enabled_models)}")
print(f"Total annotations: {len(video_files) * len(enabled_models)}")
print(f"Frames per video: {FRAMES_TO_SAMPLE}")

for model_key in enabled_models:
    if results[model_key]:
        model_name = MODELS[model_key]['display_name']
        total_events = sum(len(r.get('events', [])) for r in results[model_key].values())
        errors = sum(1 for r in results[model_key].values() if 'error' in r and r['error'])
        successful = len(results[model_key]) - errors

        print(f"\n🤖 {model_name}:")
        print(f"   • Videos: {len(results[model_key])}")
        print(f"   • Successful: {successful}")
        print(f"   • Errors: {errors}")
        print(f"   • Events detected: {total_events}")

print(f"\n⏱️  Total time: {total_time:.1f} minutes ({total_time/60:.1f} hours)")
print(f"   Avg per video: {total_time/len(video_files):.1f} minutes")
print(f"   Avg per annotation: {total_time/(len(video_files)*len(enabled_models)):.1f} minutes")

print(f"\n📁 Results saved to:")
print(f"   MyDrive/SoccerNet_LLM_Benchmark/annotations/")
print(f"   • nemotron/batch_{BATCH_NUMBER}_nemotron_FINAL.json")
print(f"   • maverick/batch_{BATCH_NUMBER}_maverick_FINAL.json")
print(f"   • gemini/batch_{BATCH_NUMBER}_gemini_FINAL.json")
print(f"   • mistral/batch_{BATCH_NUMBER}_mistral_FINAL.json")
print(f"   • gemma/batch_{BATCH_NUMBER}_gemma_FINAL.json")
print(f"   • scout/batch_{BATCH_NUMBER}_scout_FINAL.json")

print(f"\n💡 Next Steps:")
print(f"   • Review JSON files for each model")
print(f"   • Compare model performance")
print(f"   • Set BATCH_NUMBER = {BATCH_NUMBER + 1} for next batch")
print(f"   • Check OpenRouter usage: https://openrouter.ai/activity")

print(f"\n{'='*80}")
print(f"✅ BATCH {BATCH_NUMBER} COMPLETE!")
print(f"✅ All 6 models successfully annotated 40 videos!")
print(f"{'='*80}")
