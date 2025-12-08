#!/bin/bash
#SBATCH --job-name=SoccerMiniCPM
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:TitanRTX:4
#SBATCH --output=minicpm_%j.log
#SBATCH --exclusive

# ============================================================================
# MiniCPM-V 2.6 Batch Processing Script
# Usage: sbatch submit_minicpm.sh [START_VIDEO] [END_VIDEO]
#
# Examples:
#   sbatch submit_minicpm.sh 0 39     # Process all videos
#   sbatch submit_minicpm.sh 0 10     # Process first 10 videos
#   sbatch submit_minicpm.sh 20 39    # Process videos 20-39
# ============================================================================

module load python/3.12.3
module load ffmpeg
source $HOME/soccer_project/venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Set HuggingFace token
if [ -z "$HF_TOKEN" ] || [ "$HF_TOKEN" = "YOUR_TOKEN" ]; then
    echo "⚠️ WARNING: HF_TOKEN not set or is placeholder"
    echo "   Set your token: export HF_TOKEN='your_actual_token' before running sbatch"
    echo "   Continuing anyway (may fail if model requires authentication)..."
    echo ""
fi

# Parse arguments (default: 0 to 39)
START_VIDEO=${1:-0}
END_VIDEO=${2:-39}

echo "=========================================="
echo "🎬 MINICPM-V 2.6 BATCH PROCESSING"
echo "=========================================="
echo "Model: openbmb/MiniCPM-V-2_6-int4 (~7GB VRAM)"
echo "Start video: $START_VIDEO"
echo "End video: $END_VIDEO"
echo "Settings: 5 FPS, 3s window (15 frames), 2s step"
echo "Time: $(date)"
echo "=========================================="
echo ""

# Loop from START_VIDEO to END_VIDEO
for ((i=$START_VIDEO; i<=$END_VIDEO; i++))
do
    echo "----------------------------------------------------------------"
    echo "🎬 PROCESSING VIDEO INDEX: $i ($(($i - $START_VIDEO + 1))/$(($END_VIDEO - $START_VIDEO + 1)))"
    echo "----------------------------------------------------------------"

    # Check if this video already completed successfully
    MERGED_FILE="$HOME/soccer_project/final_predictions_minicpm_video_${i}.json"
    if [ -f "$MERGED_FILE" ]; then
        echo "⏭️ Skipping video $i - already processed (merged file exists)"
        echo "   File: $(basename $MERGED_FILE)"
        echo "   To reprocess: rm $MERGED_FILE"
        echo ""
        continue
    fi

    # Clean up any partial results from previous interrupted run
    PARTIAL_COUNT=$(ls $HOME/soccer_project/partial_results_minicpm_vid${i}_gpu*.json 2>/dev/null | wc -l)
    if [ $PARTIAL_COUNT -gt 0 ]; then
        echo "⚠️ Found $PARTIAL_COUNT partial file(s) from previous run - cleaning up..."
        rm $HOME/soccer_project/partial_results_minicpm_vid${i}_gpu*.json
        echo "   ✓ Cleaned"
    fi

    # Launch GPU swarm for this video
    echo "🚀 Launching 4-GPU MiniCPM-V 2.6 swarm..."
    python -u $HOME/soccer_project/minicpm_split_worker.py --gpu_id 0 --num_workers 4 --video_index $i &
    python -u $HOME/soccer_project/minicpm_split_worker.py --gpu_id 1 --num_workers 4 --video_index $i &
    python -u $HOME/soccer_project/minicpm_split_worker.py --gpu_id 2 --num_workers 4 --video_index $i &
    python -u $HOME/soccer_project/minicpm_split_worker.py --gpu_id 3 --num_workers 4 --video_index $i &

    # Wait for all 4 workers to complete
    wait

    # Merge results for this video
    echo "🔗 Merging results..."
    python $HOME/soccer_project/merge_results_minicpm.py --video_index $i

    # Verify merge succeeded
    if [ -f "$MERGED_FILE" ]; then
        # Check if file has events
        EVENT_COUNT=$(python -c "import json; print(len(json.load(open('$MERGED_FILE'))))" 2>/dev/null || echo "0")

        if [ "$EVENT_COUNT" -gt 0 ]; then
            echo "✅ Merge successful - cleaning up partial files"
            rm $HOME/soccer_project/partial_results_minicpm_vid${i}_gpu*.json 2>/dev/null
            echo "   📊 Final: $EVENT_COUNT events detected"
        else
            echo "⚠️ Merge created empty file - keeping partial files for debugging"
            echo "   This video may need manual investigation"
        fi
    else
        echo "❌ Merge failed - keeping partial files for debugging"
        echo "   Continuing to next video..."
    fi

    echo "✅ Video $i complete ($(date))"
    echo ""
done

echo "=========================================="
echo "🎉 MINICPM-V 2.6 BATCH PROCESSING COMPLETE"
echo "=========================================="
echo "Processed videos: $START_VIDEO to $END_VIDEO"
echo "Time: $(date)"
echo "=========================================="
