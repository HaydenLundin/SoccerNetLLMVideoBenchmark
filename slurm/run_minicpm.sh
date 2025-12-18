#!/bin/bash
# ============================================================================
# MiniCPM-V 2.6 Direct Execution Script (No SLURM)
# Usage: ./run_minicpm.sh [VIDEO_INDEX]
#
# Examples:
#   ./run_minicpm.sh        # Process video 0
#   ./run_minicpm.sh 5      # Process video 5
# ============================================================================

# Parse arguments
VIDEO_INDEX=${1:-0}

echo "=========================================="
echo "🎬 MINICPM-V 2.6 PROCESSING"
echo "=========================================="
echo "Model: openbmb/MiniCPM-V-2_6-int4 (~7GB VRAM)"
echo "Video index: $VIDEO_INDEX"
echo "Settings: 5 FPS, 3s window (15 frames), 2s step"
echo "Time: $(date)"
echo "=========================================="
echo ""

# Check if this video already completed successfully
MERGED_FILE="$HOME/soccer_project/final_predictions_minicpm_video_${VIDEO_INDEX}.json"
if [ -f "$MERGED_FILE" ]; then
    echo "⏭️ Video $VIDEO_INDEX already processed (merged file exists)"
    echo "   File: $(basename $MERGED_FILE)"
    echo "   To reprocess: rm $MERGED_FILE"
    exit 0
fi

# Clean up any partial results from previous interrupted run
PARTIAL_COUNT=$(ls $HOME/soccer_project/partial_results_minicpm_vid${VIDEO_INDEX}_gpu*.json 2>/dev/null | wc -l)
if [ $PARTIAL_COUNT -gt 0 ]; then
    echo "⚠️ Found $PARTIAL_COUNT partial file(s) from previous run - cleaning up..."
    rm $HOME/soccer_project/partial_results_minicpm_vid${VIDEO_INDEX}_gpu*.json
    echo "   ✓ Cleaned"
fi

# Launch GPU swarm for this video
echo "🚀 Launching 4-GPU MiniCPM-V 2.6 swarm..."
python -u $HOME/soccer_project/minicpm_split_worker.py --gpu_id 0 --num_workers 4 --video_index $VIDEO_INDEX &
python -u $HOME/soccer_project/minicpm_split_worker.py --gpu_id 1 --num_workers 4 --video_index $VIDEO_INDEX &
python -u $HOME/soccer_project/minicpm_split_worker.py --gpu_id 2 --num_workers 4 --video_index $VIDEO_INDEX &
python -u $HOME/soccer_project/minicpm_split_worker.py --gpu_id 3 --num_workers 4 --video_index $VIDEO_INDEX &

# Wait for all 4 workers to complete
wait

# Merge results for this video
echo "🔗 Merging results..."
python $HOME/soccer_project/merge_results_minicpm.py --video_index $VIDEO_INDEX

# Verify merge succeeded
if [ -f "$MERGED_FILE" ]; then
    # Check if file has events
    EVENT_COUNT=$(python -c "import json; print(len(json.load(open('$MERGED_FILE'))))" 2>/dev/null || echo "0")

    if [ "$EVENT_COUNT" -gt 0 ]; then
        echo "✅ Merge successful - cleaning up partial files"
        rm $HOME/soccer_project/partial_results_minicpm_vid${VIDEO_INDEX}_gpu*.json 2>/dev/null
        echo "   📊 Final: $EVENT_COUNT events detected"
    else
        echo "⚠️ Merge created empty file - keeping partial files for debugging"
    fi
else
    echo "❌ Merge failed - keeping partial files for debugging"
fi

echo ""
echo "=========================================="
echo "🎉 MINICPM-V 2.6 PROCESSING COMPLETE"
echo "=========================================="
echo "Video index: $VIDEO_INDEX"
echo "Time: $(date)"
echo "=========================================="
