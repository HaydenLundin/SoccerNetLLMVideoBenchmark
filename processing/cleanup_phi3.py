#!/usr/bin/env python3
"""
Cleanup Script for Incomplete Phi-3.5-Vision Video Processing

Usage:
    python cleanup_phi3.py --video_index 5
    python cleanup_phi3.py --video_index 5 --dry-run
    python cleanup_phi3.py --video_index 5 --force
"""

import os
import glob
import argparse
import json

# Configuration
PROJECT_DIR = os.path.expanduser("~/soccer_project")

parser = argparse.ArgumentParser(description="Clean up incomplete Phi-3.5-Vision video processing results")
parser.add_argument("--video_index", type=int, required=True, help="Video index to clean up")
parser.add_argument("--dry-run", action="store_true", help="Show what would be deleted without deleting")
parser.add_argument("--force", action="store_true", help="Delete even if merged file exists")
args = parser.parse_args()

print(f"🧹 Cleanup for Phi-3.5-Vision Video Index {args.video_index}")
print(f"   Project dir: {PROJECT_DIR}")
if args.dry_run:
    print("   🔍 DRY RUN MODE - Nothing will be deleted")
print()

# File patterns
partial_pattern = os.path.join(PROJECT_DIR, f"partial_results_phi3_vid{args.video_index}_gpu*.json")
merged_file = os.path.join(PROJECT_DIR, f"final_predictions_phi3_video_{args.video_index}.json")

# Find files
partial_files = sorted(glob.glob(partial_pattern))
merged_exists = os.path.exists(merged_file)

# Report status
print(f"📊 Current status:")
print(f"   Partial files: {len(partial_files)}/4 GPUs")
for f in partial_files:
    try:
        with open(f, 'r') as json_file:
            data = json.load(json_file)
            print(f"      ✓ {os.path.basename(f)}: {len(data)} events")
    except Exception as e:
        print(f"      ⚠️ {os.path.basename(f)}: ERROR - {e}")

print(f"   Merged file: {'✓ EXISTS' if merged_exists else '✗ NOT FOUND'}")
print()

# Safety check
if merged_exists and not args.force:
    print("⚠️ SAFETY WARNING:")
    print(f"   Merged file already exists: {os.path.basename(merged_file)}")
    print(f"   This video may have completed successfully.")
    print()
    print("   Options:")
    print("   1. If video completed: No cleanup needed")
    print("   2. If partial/corrupted: Use --force to delete and reprocess")
    print()
    exit(0)

# Determine action
if not partial_files and not merged_exists:
    print("✅ No files found - nothing to clean up!")
    exit(0)

# Show what will be deleted
print(f"🗑️ Files to delete:")
for f in partial_files:
    print(f"   - {os.path.basename(f)}")
if merged_exists:
    print(f"   - {os.path.basename(merged_file)} (--force specified)")
print()

# Execute deletion
if args.dry_run:
    print("✅ Dry run complete - no files deleted")
else:
    deleted_count = 0
    for f in partial_files:
        try:
            os.remove(f)
            print(f"   ✓ Deleted {os.path.basename(f)}")
            deleted_count += 1
        except Exception as e:
            print(f"   ✗ Failed to delete {os.path.basename(f)}: {e}")

    if merged_exists and args.force:
        try:
            os.remove(merged_file)
            print(f"   ✓ Deleted {os.path.basename(merged_file)}")
            deleted_count += 1
        except Exception as e:
            print(f"   ✗ Failed to delete {os.path.basename(merged_file)}: {e}")

    print()
    print(f"✅ Cleanup complete - deleted {deleted_count} file(s)")
    print(f"   Ready to reprocess video {args.video_index}")
