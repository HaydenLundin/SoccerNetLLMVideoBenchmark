#!/bin/bash
#SBATCH --job-name=SoccerRestart
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00            # Increased to 6 hours for safety
#SBATCH --gres=gpu:TitanRTX:4
#SBATCH --output=swarm_restart_%j.log
#SBATCH --exclusive

# ============================================================================
# Restart Script for Soccer Video Processing
# Usage: sbatch submit_restart.sh [START_VIDEO] [END_VIDEO]
#
# Examples:
#   sbatch submit_restart.sh 17 39    # Resume from video 17 to end
#   sbatch submit_restart.sh 17 20    # Process only videos 17-20
#   sbatch submit_restart.sh 17       # Resume from 17 to 39 (default)
# ============================================================================

module load python/3.12.3
module load ffmpeg
source $HOME/soccer_project/venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_TOKEN="YOUR_TOKEN"

# Parse arguments (default: 17 to 39)
START_VIDEO=${1:-17}
END_VIDEO=${2:-39}

echo "=========================================="
echo "🔄 RESTARTING BATCH PROCESSING"
echo "=========================================="
echo "Start video: $START_VIDEO"
echo "End video: $END_VIDEO"
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
    MERGED_FILE="$HOME/soccer_project/final_predictions_qwen7b_video_${i}.json"
    if [ -f "$MERGED_FILE" ]; then
        echo "⏭️ Skipping video $i - already processed (merged file exists)"
        echo "   File: $(basename $MERGED_FILE)"
        echo "   To reprocess: rm $MERGED_FILE"
        echo ""
        continue
    fi

    # Clean up any partial results from previous interrupted run
    PARTIAL_COUNT=$(ls $HOME/soccer_project/partial_results_qwen7b_vid${i}_gpu*.json 2>/dev/null | wc -l)
    if [ $PARTIAL_COUNT -gt 0 ]; then
        echo "⚠️ Found $PARTIAL_COUNT partial file(s) from previous run - cleaning up..."
        rm $HOME/soccer_project/partial_results_qwen7b_vid${i}_gpu*.json
        echo "   ✓ Cleaned"
    fi

    # Launch GPU swarm for this video
    echo "🚀 Launching 4-GPU swarm..."
    python -u $HOME/soccer_project/qwen_split_worker.py --gpu_id 0 --num_workers 4 --video_index $i &
    python -u $HOME/soccer_project/qwen_split_worker.py --gpu_id 1 --num_workers 4 --video_index $i &
    python -u $HOME/soccer_project/qwen_split_worker.py --gpu_id 2 --num_workers 4 --video_index $i &
    python -u $HOME/soccer_project/qwen_split_worker.py --gpu_id 3 --num_workers 4 --video_index $i &

    # Wait for all 4 workers to complete
    wait

    # Merge results for this video
    echo "🔗 Merging results..."
    python $HOME/soccer_project/merge_results.py --video_index $i

    # Verify merge succeeded
    if [ -f "$MERGED_FILE" ]; then
        echo "✅ Merge successful - cleaning up partial files"
        rm $HOME/soccer_project/partial_results_qwen7b_vid${i}_gpu*.json

        # Show event count
        EVENT_COUNT=$(python -c "import json; print(len(json.load(open('$MERGED_FILE'))))")
        echo "   📊 Final: $EVENT_COUNT events detected"
    else
        echo "❌ Merge failed - keeping partial files for debugging"
        exit 1
    fi

    echo "✅ Video $i complete ($(date))"
    echo ""
done

echo "=========================================="
echo "🎉 BATCH PROCESSING COMPLETE"
echo "=========================================="
echo "Processed videos: $START_VIDEO to $END_VIDEO"
echo "Time: $(date)"
echo "=========================================="
