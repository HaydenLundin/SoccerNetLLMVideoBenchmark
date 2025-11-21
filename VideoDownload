"""
104 EPL Games Downloader + LLM Annotation Pipeline
===================================================

WORKFLOW PER SESSION:
1. Download batch of 20 games (720p)
2. Extract audio (optional, for efficiency)
3. Run Gemini annotations
4. Run LLM #2 annotations
5. Run LLM #3 annotations
6. Benchmark against SoccerNet ground truth
7. Delete videos, keep only annotations
8. Ready for next batch!

USAGE:
- Set BATCH_NUMBER (1-5)
- Set NDA_PASSWORD
- Run and wait 8-12 hours
- All annotations saved to Drive
- Videos deleted automatically

"""

# ============================================================================
# CONFIGURATION
# ============================================================================

# Which batch are you downloading? (1, 2, 3, 4, or 5)
BATCH_NUMBER = 1

# Your NDA password
NDA_PASSWORD = "s0cc3rn3t"

# Resolution
RESOLUTION = "720p"

# LLM Processing Options
EXTRACT_AUDIO = True  # Extract audio to separate files (recommended)
RUN_GEMINI = True     # Run Gemini annotations
RUN_LLM2 = True      # Set to True when you add 2nd LLM
RUN_LLM3 = False      # Set to True when you add 3rd LLM
DELETE_VIDEOS_AFTER = True  # Delete videos after annotation (recommended)
SKIP_DOWNLOAD = False

# ============================================================================
# IMPORTS
# ============================================================================

from SoccerNet.Downloader import SoccerNetDownloader
from SoccerNet.utils import getListGames
from google.colab import drive, userdata
import google.generativeai as genai
import os
import json
import random
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_audio_from_video(video_path, output_path):
    """Extract audio from video file using ffmpeg"""
    try:
        cmd = [
            'ffmpeg', '-i', video_path,
            '-vn',  # No video
            '-acodec', 'libmp3lame',  # MP3 codec
            '-ab', '192k',  # Bitrate
            '-ar', '22050',  # Sample rate (matches your methodology)
            '-y',  # Overwrite
            output_path
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except Exception as e:
        print(f"  ⚠️  Audio extraction failed: {e}")
        return False

def extract_audio_from_batch(local_dir):
    """Extract audio from all videos in batch"""
    print(f"\n{'='*80}")
    print(f"EXTRACTING AUDIO FROM VIDEOS")
    print(f"{'='*80}")

    audio_dir = os.path.join(local_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    video_files = []
    for root, dirs, files in os.walk(local_dir):
        for file in files:
            if file.endswith('.mkv'):
                video_files.append(os.path.join(root, file))

    print(f"Found {len(video_files)} video files")

    success_count = 0
    for i, video_path in enumerate(video_files, 1):
        video_name = Path(video_path).stem
        audio_path = os.path.join(audio_dir, f"{video_name}.mp3")

        print(f"[{i}/{len(video_files)}] Extracting: {video_name}")
        if extract_audio_from_video(video_path, audio_path):
            success_count += 1
            print(f"  ✅ Saved to: {audio_path}")

    print(f"\n✅ Audio extraction complete: {success_count}/{len(video_files)} successful")
    return audio_dir

def setup_gemini():
    """Configure Gemini API"""
    try:
        api_key = userdata.get('GOOGLE_API_KEY')
        genai.configure(api_key=api_key)
        return genai.GenerativeModel('gemini-2.5-pro')
    except Exception as e:
        print(f"⚠️  Gemini setup failed: {e}")
        print("Make sure GOOGLE_API_KEY is set in Colab Secrets")
        return None

def annotate_video_with_gemini(model, video_path, game_info):
    """
    Annotate a single video with Gemini

    Args:
        model: Gemini model instance
        video_path: Path to video file
        game_info: Dict with game metadata (name, half, etc.)

    Returns:
        Dict with annotations
    """
    try:
        print(f"  📤 Uploading video to Gemini...")
        video_file = genai.upload_file(path=video_path)

        # Wait for processing
        import time
        while video_file.state.name == "PROCESSING":
            time.sleep(2)
            video_file = genai.get_file(video_file.name)

        if video_file.state.name == "FAILED":
            raise ValueError("Video processing failed")

        print(f"  🤖 Running Gemini analysis...")

        # Annotation prompt based on SoccerNet categories
        prompt = """
        Analyze this soccer game video and identify ALL events with precise timestamps.

        Detect these event types:
        1. Goals - any ball crossing the goal line
        2. Shots on/off target
        3. Fouls - any rule violations
        4. Yellow cards
        5. Red cards
        6. Substitutions - player changes
        7. Corners - corner kicks
        8. Throw-ins
        9. Penalties
        10. Offsides
        11. Clearances
        12. Passes (major/key passes only)
        13. Direct free-kicks
        14. Indirect free-kicks
        15. Kick-off
        16. Ball out of play
        17. Shots on target

        For EACH event, provide:
        - event_type: (goal, foul, card, etc.)
        - timestamp: MM:SS format from video start
        - half: which half (1 or 2)
        - team: "home" or "away" (if identifiable)
        - player: player number or name (if visible)
        - confidence: your confidence level (0.0-1.0)
        - description: brief description

        CRITICAL: Return ONLY valid JSON, no markdown, no explanation.
        Format:
        {
          "game": "game_name",
          "half": 1,
          "events": [
            {
              "event_type": "goal",
              "timestamp": "12:34",
              "team": "home",
              "player": "#10",
              "confidence": 0.95,
              "description": "Header from corner kick"
            },
            ...
          ]
        }
        """

        response = model.generate_content([video_file, prompt])
        response_text = response.text.strip()

        # Clean up response (remove markdown if present)
        if response_text.startswith('```json'):
            response_text = response_text.split('```json')[1]
        if response_text.endswith('```'):
            response_text = response_text.rsplit('```', 1)[0]
        response_text = response_text.strip()

        # Parse JSON
        annotations = json.loads(response_text)

        # Add metadata
        annotations['model'] = 'gemini-2.5-pro'
        annotations['game_info'] = game_info
        annotations['processed_at'] = datetime.now().isoformat()

        print(f"  ✅ Found {len(annotations.get('events', []))} events")

        # Cleanup
        genai.delete_file(video_file.name)

        return annotations

    except Exception as e:
        print(f"  ❌ Gemini annotation failed: {e}")
        return {
            "error": str(e),
            "game_info": game_info,
            "model": "gemini-2.5-pro",
            "events": []
        }

def annotate_batch_with_gemini(local_dir, batch_games):
    """Run Gemini annotations on all games in batch"""
    print(f"\n{'='*80}")
    print(f"RUNNING GEMINI ANNOTATIONS")
    print(f"{'='*80}")

    model = setup_gemini()
    if model is None:
        return {}

    annotations = {}
    video_files = []

    # Find all video files
    for root, dirs, files in os.walk(local_dir):
        for file in files:
            if file.endswith('.mkv'):
                video_files.append(os.path.join(root, file))

    print(f"Processing {len(video_files)} video files")

    for i, video_path in enumerate(video_files, 1):
        video_name = Path(video_path).stem
        game_path = Path(video_path).parent.parent.name + "/" + Path(video_path).parent.name

        print(f"\n[{i}/{len(video_files)}] Processing: {video_name}")
        print(f"  Game: {game_path}")

        game_info = {
            "video_file": video_name,
            "game_path": game_path,
            "half": 1 if "1st" in video_name or "1_" in video_name else 2
        }

        result = annotate_video_with_gemini(model, video_path, game_info)
        annotations[video_name] = result

    return annotations

def setup_qwen_vl():
    """Configure Qwen3-VL via OpenRouter"""
    try:
        api_key = userdata.get('openrouter')
        if not api_key:
            print("⚠️  OPENROUTER_API_KEY not found in Colab Secrets")
            return None
        return api_key
    except Exception as e:
        print(f"⚠️  Qwen VL setup failed: {e}")
        return None

def extract_frames_for_annotation(video_path, fps=0.2):
    """
    Extract frames from video for LLM processing
    fps=0.2 means 1 frame every 5 seconds

    Returns: List of frame paths
    """
    import subprocess
    from pathlib import Path

    output_dir = Path(video_path).parent / f"{Path(video_path).stem}_frames"
    output_dir.mkdir(exist_ok=True)

    output_pattern = str(output_dir / "frame_%04d.jpg")

    try:
        # Use ffmpeg to extract frames
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-vf', f'fps={fps}',  # 1 frame every 5 seconds
            '-q:v', '2',  # High quality
            '-y',
            output_pattern
        ]

        subprocess.run(cmd, check=True, capture_output=True)

        # Get list of extracted frames
        frames = sorted(output_dir.glob("frame_*.jpg"))
        print(f"  📸 Extracted {len(frames)} frames")
        return [str(f) for f in frames]

    except Exception as e:
        print(f"  ❌ Frame extraction failed: {e}")
        return []

def encode_image_to_base64(image_path):
    """Convert image to base64 for API"""
    import base64
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

def annotate_video_with_qwen_vl(video_path, game_info):
    """
    Annotate video with Qwen3-VL-8B-Thinking via OpenRouter

    Args:
        video_path: Path to video file
        game_info: Dict with game metadata

    Returns:
        Dict with annotations
    """
    try:
        import requests

        api_key = setup_qwen_vl()
        if not api_key:
            return {"error": "API key not configured", "events": []}

        print(f"  🎬 Processing video with Qwen3-VL-8B-Thinking...")

        # Extract frames (1 per 5 seconds = ~540 frames for 45-min half)
        frame_paths = extract_frames_for_annotation(video_path, fps=0.2)

        if not frame_paths:
            return {"error": "Frame extraction failed", "events": []}

        # Sample frames if too many (to avoid token limits)
        # Take every Nth frame to get ~100 frames
        if len(frame_paths) > 100:
            step = len(frame_paths) // 100
            frame_paths = frame_paths[::step]
            print(f"  📊 Sampled to {len(frame_paths)} frames for analysis")

        # Prepare prompt
        prompt = """
        Analyze these soccer game video frames and identify ALL events with precise timestamps.

        Detect these event types:
        1. Goals - any ball crossing the goal line
        2. Shots on/off target
        3. Fouls - any rule violations
        4. Yellow cards
        5. Red cards
        6. Substitutions - player changes
        7. Corners - corner kicks
        8. Throw-ins
        9. Penalties
        10. Offsides
        11. Clearances
        12. Passes (major/key passes only)
        13. Direct free-kicks
        14. Indirect free-kicks
        15. Kick-off
        16. Ball out of play
        17. Shots on target

        For EACH event you detect in the frames, provide:
        - event_type: (goal, foul, card, etc.)
        - frame_number: which frame shows the event
        - timestamp: estimate MM:SS format (frames are sampled every 5 seconds)
        - team: "home" or "away" (if identifiable)
        - player: player number or name (if visible)
        - confidence: your confidence level (0.0-1.0)
        - description: brief description

        CRITICAL: Return ONLY valid JSON, no markdown, no explanation.
        Format:
        {
          "game": "game_name",
          "half": 1,
          "total_frames_analyzed": 100,
          "events": [
            {
              "event_type": "goal",
              "frame_number": 45,
              "timestamp": "03:45",
              "team": "home",
              "player": "#10",
              "confidence": 0.95,
              "description": "Header from corner kick"
            }
          ]
        }
        """

        # Build message content with frames
        content = [{"type": "text", "text": prompt}]

        # Add frames as images (base64 encoded)
        print(f"  📤 Encoding {len(frame_paths)} frames...")
        for i, frame_path in enumerate(frame_paths):
            if i % 20 == 0:  # Progress update every 20 frames
                print(f"    Encoding frame {i+1}/{len(frame_paths)}...")

            base64_image = encode_image_to_base64(frame_path)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            })

        # OpenRouter API call
        url = "https://openrouter.ai/api/v1/chat/completions"

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://colab.research.google.com",
            "X-Title": "SoccerNet LLM Benchmark"
        }

        payload = {
            "model": "qwen/qwen3-vl-8b-thinking",
            "messages": [
                {
                    "role": "user",
                    "content": content
                }
            ],
            "max_tokens": 4000
        }

        print(f"  🤖 Sending request to Qwen3-VL via OpenRouter...")
        response = requests.post(url, headers=headers, json=payload, timeout=600)
        response.raise_for_status()

        result = response.json()
        response_text = result['choices'][0]['message']['content'].strip()

        # Clean up response
        if response_text.startswith('```json'):
            response_text = response_text.split('```json')[1]
        if response_text.startswith('```'):
            response_text = response_text.split('```')[1]
        if response_text.endswith('```'):
            response_text = response_text.rsplit('```', 1)[0]
        response_text = response_text.strip()

        # Parse JSON
        annotations = json.loads(response_text)

        # Add metadata
        annotations['model'] = 'qwen3-vl-8b-thinking'
        annotations['game_info'] = game_info
        annotations['processed_at'] = datetime.now().isoformat()
        annotations['frames_processed'] = len(frame_paths)

        print(f"  ✅ Found {len(annotations.get('events', []))} events")

        # Cleanup frames
        import shutil
        frame_dir = Path(frame_paths[0]).parent
        shutil.rmtree(frame_dir, ignore_errors=True)

        return annotations

    except Exception as e:
        print(f"  ❌ Qwen VL annotation failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            "error": str(e),
            "game_info": game_info,
            "model": "qwen3-vl-8b-thinking",
            "events": []
        }

def annotate_with_llm2(local_dir, batch_games):
    """Run Qwen3-VL annotations on all games in batch"""
    print(f"\n{'='*80}")
    print(f"RUNNING QWEN3-VL-8B-THINKING ANNOTATIONS")
    print(f"{'='*80}")

    api_key = setup_qwen_vl()
    if api_key is None:
        print("⚠️  Skipping Qwen VL annotations (API key not configured)")
        return {}

    annotations = {}
    video_files = []

    # Find all video files
    for root, dirs, files in os.walk(local_dir):
        for file in files:
            if file.endswith('.mkv'):
                video_files.append(os.path.join(root, file))

    print(f"Processing {len(video_files)} video files")

    for i, video_path in enumerate(video_files, 1):
        video_name = Path(video_path).stem
        game_path = Path(video_path).parent.parent.name + "/" + Path(video_path).parent.name

        print(f"\n[{i}/{len(video_files)}] Processing: {video_name}")
        print(f"  Game: {game_path}")

        game_info = {
            "video_file": video_name,
            "game_path": game_path,
            "half": 1 if "1st" in video_name or "1_" in video_name else 2
        }

        result = annotate_video_with_qwen_vl(video_path, game_info)
        annotations[video_name] = result

    return annotations

def annotate_with_llm3(local_dir, batch_games):
    """Placeholder for 3rd LLM - implement your own"""
    print(f"\n{'='*80}")
    print(f"RUNNING LLM #3 ANNOTATIONS")
    print(f"{'='*80}")
    print("⚠️  LLM #3 not implemented yet - add your code here")
    return {}

def load_soccernet_labels(local_dir):
    """Load SoccerNet ground truth labels"""
    print(f"\n{'='*80}")
    print(f"LOADING SOCCERNET GROUND TRUTH LABELS")
    print(f"{'='*80}")

    labels = {}
    label_files = []

    for root, dirs, files in os.walk(local_dir):
        for file in files:
            if file == 'Labels-v2.json':
                label_files.append(os.path.join(root, file))

    print(f"Found {len(label_files)} label files")

    for label_path in label_files:
        game_path = Path(label_path).parent.parent.name + "/" + Path(label_path).parent.name
        try:
            with open(label_path, 'r') as f:
                labels[game_path] = json.load(f)
            print(f"  ✅ Loaded: {game_path}")
        except Exception as e:
            print(f"  ❌ Failed to load {game_path}: {e}")

    return labels

def benchmark_annotations(gemini_annotations, llm2_annotations, llm3_annotations, ground_truth):
    """Compare LLM annotations against SoccerNet ground truth"""
    print(f"\n{'='*80}")
    print(f"BENCHMARKING ANNOTATIONS")
    print(f"{'='*80}")

    results = {
        "batch_number": BATCH_NUMBER,
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "gemini": {"total_events": 0, "avg_confidence": 0.0},
            "llm2": {"total_events": 0, "avg_confidence": 0.0},
            "llm3": {"total_events": 0, "avg_confidence": 0.0},
            "ground_truth": {"total_events": 0}
        },
        "detailed_comparison": {}
    }

    # Count events from each source
    for video_name, annotations in gemini_annotations.items():
        events = annotations.get('events', [])
        results["summary"]["gemini"]["total_events"] += len(events)
        if events:
            avg_conf = sum(e.get('confidence', 0.0) for e in events) / len(events)
            results["summary"]["gemini"]["avg_confidence"] = avg_conf

    # Count ground truth events
    for game, labels in ground_truth.items():
        if 'annotations' in labels:
            results["summary"]["ground_truth"]["total_events"] += len(labels['annotations'])

    print(f"\nEvent Counts:")
    print(f"  Gemini:       {results['summary']['gemini']['total_events']} events")
    print(f"  Ground Truth: {results['summary']['ground_truth']['total_events']} events")
    print(f"  Coverage:     {results['summary']['gemini']['total_events'] / max(results['summary']['ground_truth']['total_events'], 1) * 100:.1f}%")

    # TODO: Add detailed metrics (precision, recall, F1) - implement your own
    # This would require temporal matching of events (within N seconds)
    print(f"\n⚠️  Detailed metrics (precision/recall/F1) not implemented yet")
    print(f"   Add your own comparison logic based on your research needs")

    return results

def save_to_drive(data, filename, subfolder=""):
    """Save data to Google Drive"""
    drive_base = "/content/drive/MyDrive/SoccerNet_LLM_Benchmark"
    if subfolder:
        drive_path = os.path.join(drive_base, subfolder)
    else:
        drive_path = drive_base

    os.makedirs(drive_path, exist_ok=True)

    filepath = os.path.join(drive_path, filename)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"  💾 Saved: {filepath}")
    return filepath

def cleanup_videos(local_dir):
    """Delete video files to free space"""
    print(f"\n{'='*80}")
    print(f"CLEANING UP VIDEOS")
    print(f"{'='*80}")

    video_count = 0
    freed_gb = 0

    for root, dirs, files in os.walk(local_dir):
        for file in files:
            if file.endswith('.mkv'):
                filepath = os.path.join(root, file)
                size_gb = os.path.getsize(filepath) / (1024**3)
                os.remove(filepath)
                video_count += 1
                freed_gb += size_gb

    print(f"✅ Deleted {video_count} video files")
    print(f"✅ Freed {freed_gb:.1f} GB of space")

    # Show remaining disk space
    disk_info = shutil.disk_usage("/content")
    free_gb = disk_info.free / (1024**3)
    print(f"💾 Free disk space: {free_gb:.1f} GB")

# ============================================================================
# MAIN SCRIPT
# ============================================================================

print("="*80)
print(f"104 EPL GAMES DOWNLOADER + LLM ANNOTATION - BATCH {BATCH_NUMBER}/5")
print("="*80)

# Check password
if NDA_PASSWORD == "YOUR_PASSWORD_HERE":
    print("❌ ERROR: Set your NDA_PASSWORD first!")
    exit()

# Mount Google Drive
print("\n📁 Mounting Google Drive...")
drive.mount('/content/drive')
print("✅ Mounted!")

# Get all EPL games
print("\n🔍 Finding EPL games...")
all_epl_games = []
for split in ["train", "valid", "test", "challenge"]:
    games = getListGames(split=split)
    epl_games = [g for g in games if g.startswith('england_epl/')]
    all_epl_games.extend(epl_games)

print(f"Found {len(all_epl_games)} total EPL games")

# Use ALL 104 games (no sampling)
selected_104_games = all_epl_games

# Save the complete list to Drive for reference
games_list_file = "/content/drive/MyDrive/selected_104_epl_games.json"
if not os.path.exists(games_list_file):
    with open(games_list_file, 'w') as f:
        json.dump(selected_104_games, f, indent=2)
    print(f"✅ Saved game list to Drive")
else:
    with open(games_list_file, 'r') as f:
        selected_104_games = json.load(f)
    print(f"✅ Loaded existing game list from Drive")

# Divide into 5 batches
batch_1 = selected_104_games[0:20]
batch_2 = selected_104_games[20:40]
batch_3 = selected_104_games[40:60]
batch_4 = selected_104_games[60:80]
batch_5 = selected_104_games[80:104]

batches = {1: batch_1, 2: batch_2, 3: batch_3, 4: batch_4, 5: batch_5}
current_batch = batches[BATCH_NUMBER]

print(f"\n📦 Batch {BATCH_NUMBER} contains {len(current_batch)} games")
print(f"Resolution: {RESOLUTION}")

# Set up directories
local_dir = f"/content/soccernet_batch_{BATCH_NUMBER}"
drive_dir = f"/content/drive/MyDrive/SoccerNet_EPL/batch_{BATCH_NUMBER}"

os.makedirs(local_dir, exist_ok=True)
os.makedirs(drive_dir, exist_ok=True)

# ============================================================================
# STEP 1: DOWNLOAD VIDEOS
# ============================================================================

if SKIP_DOWNLOAD and os.path.exists(local_dir):
    print(f"\n{'='*80}")
    print(f"⏭️  SKIPPING DOWNLOAD - Videos already exist")
    print(f"{'='*80}")

    # Count existing videos
    video_files = []
    for root, dirs, files in os.walk(local_dir):
        for file in files:
            if file.endswith('.mkv'):
                video_files.append(file)

    successful = len(video_files)
    failed = 0

    print(f"✅ Found {successful} existing video files in {local_dir}")
    print(f"Proceeding to annotation step...\n")
else:
  print(f"\n⬇️  Starting downloads to {local_dir}")
  downloader = SoccerNetDownloader(LocalDirectory=local_dir)
  downloader.password = NDA_PASSWORD

  if RESOLUTION == "720p":
      files_to_download = ["1_720p.mkv", "2_720p.mkv", "Labels-v2.json"]
  else:
      files_to_download = ["1_224p.mkv", "2_224p.mkv", "Labels-v2.json"]

  print(f"\n{'='*80}")
  print(f"STEP 1: DOWNLOADING {len(current_batch)} GAMES")
  print(f"{'='*80}\n")

  successful = 0
  failed = 0

  for i, game in enumerate(current_batch, 1):
      try:
          print(f"[{i}/{len(current_batch)}] {game}")
          downloader.downloadGame(files=files_to_download, game=game)
          successful += 1
          print(f"  ✅ Downloaded\n")
      except Exception as e:
          failed += 1
          print(f"  ❌ Failed: {e}\n")

  print(f"\n{'='*80}")
  print(f"DOWNLOAD COMPLETE")
  print(f"{'='*80}")
  print(f"✅ Successful: {successful}")
  print(f"❌ Failed: {failed}")

# ============================================================================
# STEP 2: EXTRACT AUDIO (OPTIONAL)
# ============================================================================

audio_dir = None
if EXTRACT_AUDIO and successful > 0:
    audio_dir = extract_audio_from_batch(local_dir)

# ============================================================================
# Step 1.5: Copy Videos to Drive Immediately
# ============================================================================

if successful > 0:
    print(f"\n{'='*80}")
    print(f"STEP 1.5: BACKING UP VIDEOS TO GOOGLE DRIVE")
    print(f"{'='*80}")

    drive_backup_dir = f"/content/drive/MyDrive/SoccerNet_EPL/batch_{BATCH_NUMBER}"
    os.makedirs(drive_backup_dir, exist_ok=True)

    print(f"📤 Copying videos to: {drive_backup_dir}")
    print(f"This may take 15-30 minutes for 160 GB...")

    try:
        import shutil
        # Copy entire directory
        for item in os.listdir(local_dir):
            src = os.path.join(local_dir, item)
            dst = os.path.join(drive_backup_dir, item)

            if os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
                print(f"  ✅ Copied folder: {item}")
            else:
                shutil.copy2(src, dst)
                print(f"  ✅ Copied file: {item}")

        print(f"\n✅ All videos backed up to Google Drive!")
        print(f"📁 Location: {drive_backup_dir}")

    except Exception as e:
        print(f"⚠️ Error copying to Drive: {e}")
        print(f"Continuing anyway - videos are still in local storage")

# ============================================================================
# STEP 3: LOAD GROUND TRUTH LABELS
# ============================================================================

ground_truth_labels = load_soccernet_labels(local_dir)

# ============================================================================
# STEP 4: RUN LLM ANNOTATIONS
# ============================================================================

gemini_annotations = {}
llm2_annotations = {}
llm3_annotations = {}

if RUN_GEMINI and successful > 0:
    gemini_annotations = annotate_batch_with_gemini(local_dir, current_batch)
    save_to_drive(
        gemini_annotations,
        f"batch_{BATCH_NUMBER}_annotations.json",
        "annotations/gemini"
    )

if RUN_LLM2 and successful > 0:
    llm2_annotations = annotate_with_llm2(local_dir, current_batch)
    save_to_drive(
        llm2_annotations,
        f"batch_{BATCH_NUMBER}_annotations.json",
        "annotations/llm2"
    )

if RUN_LLM3 and successful > 0:
    llm3_annotations = annotate_with_llm3(local_dir, current_batch)
    save_to_drive(
        llm3_annotations,
        f"batch_{BATCH_NUMBER}_annotations.json",
        "annotations/llm3"
    )

# ============================================================================
# STEP 5: BENCHMARK RESULTS
# ============================================================================

if successful > 0:
    benchmark_results = benchmark_annotations(
        gemini_annotations,
        llm2_annotations,
        llm3_annotations,
        ground_truth_labels
    )

    save_to_drive(
        benchmark_results,
        f"batch_{BATCH_NUMBER}_benchmark.json",
        "benchmarks"
    )

    # Save ground truth for reference
    save_to_drive(
        ground_truth_labels,
        f"batch_{BATCH_NUMBER}_ground_truth.json",
        "ground_truth"
    )

# ============================================================================
# STEP 6: CLEANUP
# ============================================================================

if DELETE_VIDEOS_AFTER and successful > 0:
    cleanup_videos(local_dir)
else:
    print(f"\n⚠️  Videos kept (DELETE_VIDEOS_AFTER=False)")
    print(f"📁 Location: {local_dir}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print(f"\n{'='*80}")
if BATCH_NUMBER < 5:
    print(f"✅ BATCH {BATCH_NUMBER} COMPLETE!")
    print(f"📝 Next: Set BATCH_NUMBER = {BATCH_NUMBER + 1} and run again")
else:
    print(f"🎉 ALL 104 EPL GAMES PROCESSED!")
    print(f"📁 All annotations saved to: /content/drive/MyDrive/SoccerNet_LLM_Benchmark/")
print(f"{'='*80}")
